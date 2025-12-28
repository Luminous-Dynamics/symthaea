# 🎙️ Week 13 Day 2: Kokoro TTS Integration Plan (Phase 2)

**Date**: December 10, 2025
**Status**: 📋 Planning Complete - Ready for Implementation
**Goal**: Full Kokoro-82M TTS integration with emotional prosody

---

## 🎯 Objective

Replace the Phase 1 sine wave audio generation with real Kokoro-82M TTS model, maintaining all prosody modulation capabilities while adding natural-sounding speech.

---

## 📊 Phase 1 Achievements (Completed)

✅ **Real Audio Generation**: Sine waves with prosody
✅ **WAV Export**: Save and load audio files
✅ **Prosody System**: Pitch, speed, energy modulation
✅ **Testing**: 9/9 tests passing, 200/200 total project tests
✅ **Architecture**: Clean, extensible voice synthesis system

---

## 🔬 Kokoro-82M Model Details

### Model Information
- **Name**: Kokoro-82M
- **Parameters**: 82 million
- **Source**: HuggingFace Hub - `hexgrad/Kokoro-82M`
- **Format**: ONNX (optimized neural network)
- **Size**: ~50MB model file
- **Sample Rate**: 24kHz
- **Inference**: CPU-friendly, <100ms target

### Required Files
```
models/kokoro-82m/
├── model.onnx          # Main TTS model (~50MB)
├── config.json         # Model configuration
├── tokenizer.json      # Text tokenization
└── phonemes.txt        # Phoneme mappings (if needed)
```

### Download Strategy
```rust
// Use hf-hub crate for automatic downloading
use hf_hub::api::sync::Api;

pub fn download_kokoro_model(cache_dir: &Path) -> Result<PathBuf> {
    let api = Api::new()?;
    let repo = api.model("hexgrad/Kokoro-82M".to_string());

    // Download required files
    let model_path = repo.get("model.onnx")?;
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    Ok(model_path)
}
```

---

## 🛠️ Implementation Plan

### Step 1: Model Download and Caching
**Goal**: Download Kokoro model on first use, cache for subsequent uses

```rust
impl LarynxActor {
    fn ensure_model_downloaded(&mut self) -> Result<PathBuf> {
        let cache_dir = self.config.model_cache_dir.clone()
            .unwrap_or_else(|| PathBuf::from("models/kokoro-82m"));

        let model_path = cache_dir.join("model.onnx");

        if !model_path.exists() {
            println!("📥 Downloading Kokoro-82M model (~50MB)...");
            std::fs::create_dir_all(&cache_dir)?;
            download_kokoro_model(&cache_dir)?;
            println!("✅ Model downloaded successfully!");
        }

        Ok(model_path)
    }
}
```

**Test**: `test_model_download_and_cache`

### Step 2: ONNX Session Initialization
**Goal**: Load model into ONNX Runtime for inference

```rust
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};

impl LarynxActor {
    fn load_onnx_model(&mut self, model_path: &Path) -> Result<()> {
        let environment = Environment::builder()
            .with_name("kokoro-tts")
            .build()?;

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;

        self.onnx_session = Some(session);
        Ok(())
    }
}
```

**Test**: `test_onnx_model_loading`

### Step 3: Text Tokenization
**Goal**: Convert input text to token IDs for the model

```rust
use tokenizers::Tokenizer;

impl LarynxActor {
    fn tokenize_text(&self, text: &str) -> Result<Vec<i64>> {
        let tokenizer = Tokenizer::from_file(&self.tokenizer_path)?;

        // Tokenize text
        let encoding = tokenizer.encode(text, false)?;
        let token_ids: Vec<i64> = encoding.get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        Ok(token_ids)
    }
}
```

**Test**: `test_text_tokenization`

### Step 4: ONNX Inference
**Goal**: Run model inference to generate raw audio

```rust
impl LarynxActor {
    fn run_inference(&self, token_ids: Vec<i64>) -> Result<Vec<f32>> {
        let session = self.onnx_session.as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;

        // Prepare input tensor
        let input_shape = vec![1, token_ids.len()];
        let input_tensor = Value::from_array(
            session.allocator(),
            &token_ids
        )?;

        // Run inference
        let outputs = session.run(vec![input_tensor])?;

        // Extract audio from output
        let audio_tensor = &outputs[0];
        let audio: Vec<f32> = audio_tensor.try_extract()?.view().to_slice()?.to_vec();

        Ok(audio)
    }
}
```

**Test**: `test_onnx_inference`

### Step 5: Prosody Application to Real Speech
**Goal**: Apply pitch, speed, and energy modulation to TTS output

This is the **critical innovation** - applying prosody to real speech!

```rust
impl LarynxActor {
    fn apply_prosody_to_audio(&self, audio: Vec<f32>, prosody: Prosody) -> Result<Vec<f32>> {
        let sample_rate = self.config.sample_rate as f32;

        // 1. Speed modulation via resampling
        let speed_adjusted = if (prosody.speed - 1.0).abs() > 0.01 {
            self.resample_audio(&audio, prosody.speed)?
        } else {
            audio
        };

        // 2. Pitch shifting via phase vocoder or simple approach
        let pitch_adjusted = if (prosody.pitch - 1.0).abs() > 0.01 {
            self.shift_pitch(&speed_adjusted, prosody.pitch, sample_rate)?
        } else {
            speed_adjusted
        };

        // 3. Energy modulation (amplitude scaling)
        let energy_adjusted: Vec<f32> = pitch_adjusted
            .iter()
            .map(|&sample| sample * prosody.energy)
            .collect();

        Ok(energy_adjusted)
    }

    fn resample_audio(&self, audio: &[f32], speed: f32) -> Result<Vec<f32>> {
        // Linear interpolation resampling
        // For speed = 1.5 (faster), we sample every 1.5th sample
        // For speed = 0.8 (slower), we interpolate more samples

        let new_len = (audio.len() as f32 / speed) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let pos = (i as f32) * speed;
            let idx = pos.floor() as usize;
            let frac = pos - pos.floor();

            if idx + 1 < audio.len() {
                // Linear interpolation
                let sample = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac;
                resampled.push(sample);
            } else if idx < audio.len() {
                resampled.push(audio[idx]);
            }
        }

        Ok(resampled)
    }

    fn shift_pitch(&self, audio: &[f32], pitch: f32, sample_rate: f32) -> Result<Vec<f32>> {
        // Simple pitch shifting approach:
        // 1. Resample to change frequency (affects both speed and pitch)
        // 2. Then resample back to original duration (corrects speed)

        // For pitch = 1.2 (higher), resample to 1.2x speed
        let pitch_shifted = self.resample_audio(audio, pitch)?;

        // Resample back to original length
        let target_len = audio.len();
        let current_len = pitch_shifted.len();
        let correction = current_len as f32 / target_len as f32;

        self.resample_audio(&pitch_shifted, correction)
    }
}
```

**Tests**:
- `test_prosody_speed_modulation`
- `test_prosody_pitch_shifting`
- `test_prosody_energy_scaling`
- `test_full_prosody_pipeline`

### Step 6: Integration with Existing `speak()` Method
**Goal**: Replace sine wave generation with TTS + prosody

```rust
impl LarynxActor {
    pub async fn speak(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Calculate prosody based on endocrine state
        let prosody = self.calculate_prosody().await;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_utterances += 1;
        stats.total_characters += text.len() as u64;
        stats.current_prosody = prosody;
        drop(stats);

        // Use TTS if model is loaded, otherwise fallback to sine wave
        let audio = if self.onnx_session.is_some() {
            // Real TTS pipeline
            let token_ids = self.tokenize_text(text)?;
            let raw_audio = self.run_inference(token_ids)?;
            self.apply_prosody_to_audio(raw_audio, prosody)?
        } else {
            // Fallback to Phase 1 sine wave (for testing/development)
            self.generate_sine_wave_audio(text, prosody)?
        };

        // Update statistics
        let elapsed = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.total_ms_spent += elapsed.as_millis() as u64;
        stats.total_atp_spent += 5.0;

        Ok(audio)
    }

    // Keep Phase 1 implementation as fallback
    fn generate_sine_wave_audio(&self, text: &str, prosody: Prosody) -> Result<Vec<f32>> {
        // [Current Phase 1 implementation]
        // ...
    }
}
```

**Test**: `test_speak_with_kokoro` (verifies TTS path is used when model loaded)

---

## 🧪 Testing Strategy

### Test Hierarchy
1. **Unit Tests** (individual components)
   - Model download
   - Tokenization
   - ONNX inference
   - Prosody transformations

2. **Integration Tests** (full pipeline)
   - End-to-end TTS with prosody
   - Fallback to sine wave when model unavailable
   - Performance benchmarks

3. **Comparison Tests** (quality verification)
   - Compare stressed vs calm speech
   - Verify prosody affects output
   - Measure inference latency

### New Tests to Add
```rust
#[tokio::test]
async fn test_kokoro_model_loading() {
    // Verify model can be loaded from cache
}

#[tokio::test]
async fn test_kokoro_inference_speed() {
    // Ensure inference < 100ms for typical utterance
}

#[tokio::test]
async fn test_prosody_on_real_speech() {
    // Verify stressed vs calm speech has measurable differences
    let config = LarynxConfig::default();
    let mut larynx = LarynxActor::new(config).unwrap();

    // Load model (will download if needed)
    larynx.ensure_model_loaded().await.unwrap();

    // Generate stressed speech
    let stressed_endocrine = EndocrineState {
        cortisol: 0.9,
        dopamine: 0.3,
        ..Default::default()
    };
    let stressed_audio = larynx.speak_with_endocrine("Hello", Some(&stressed_endocrine)).await.unwrap();

    // Generate calm speech
    let calm_endocrine = EndocrineState {
        cortisol: 0.1,
        dopamine: 0.7,
        ..Default::default()
    };
    let calm_audio = larynx.speak_with_endocrine("Hello", Some(&calm_endocrine)).await.unwrap();

    // Verify they're different
    let difference = stressed_audio.iter()
        .zip(calm_audio.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / stressed_audio.len() as f32;

    assert!(difference > 0.01, "Prosody should affect real speech");
}

#[tokio::test]
async fn test_fallback_to_sine_wave() {
    // Verify system works even without model
}
```

---

## 📊 Success Criteria

### Functional Requirements ✅
- [ ] Model downloads automatically on first use
- [ ] ONNX inference completes successfully
- [ ] Prosody modulation works on real speech
- [ ] All existing tests still pass
- [ ] New tests for TTS integration pass

### Performance Requirements 🚀
- [ ] Model download: < 2 minutes (50MB)
- [ ] Model loading: < 1 second
- [ ] Inference time: < 100ms per utterance
- [ ] Prosody application: < 50ms
- [ ] Total latency: < 200ms end-to-end

### Quality Requirements 🎯
- [ ] Natural-sounding speech (subjective evaluation)
- [ ] Prosody clearly audible (stressed vs calm)
- [ ] No audio artifacts from prosody transformations
- [ ] WAV files play correctly in audio players
- [ ] Emotional states distinguishable by ear

---

## 🚧 Implementation Phases

### Phase 2a: Model Integration (1-2 hours)
- Implement model download
- Add ONNX session loading
- Test model initialization
- **Deliverable**: Model loads successfully

### Phase 2b: Inference Pipeline (2-3 hours)
- Implement tokenization
- Add ONNX inference
- Test raw TTS output
- **Deliverable**: TTS generates audio

### Phase 2c: Prosody Application (2-3 hours)
- Implement speed resampling
- Add pitch shifting
- Apply energy scaling
- **Deliverable**: Prosody works on real speech

### Phase 2d: Integration and Testing (1-2 hours)
- Integrate with `speak()` method
- Add comprehensive tests
- Benchmark performance
- **Deliverable**: Full system working

**Total Estimated Time**: 6-10 hours

---

## 🔄 Fallback Strategy

If Kokoro integration encounters issues:
1. **Phase 1 remains working** - Sine wave audio always available
2. **Graceful degradation** - System detects missing model and uses fallback
3. **Clear error messages** - Users know if TTS is unavailable
4. **Testability preserved** - All tests can run without model

```rust
impl LarynxActor {
    pub fn is_tts_available(&self) -> bool {
        self.onnx_session.is_some()
    }

    pub fn get_synthesis_mode(&self) -> &str {
        if self.is_tts_available() {
            "Kokoro TTS"
        } else {
            "Sine Wave (Fallback)"
        }
    }
}
```

---

## 📚 Dependencies

### Required Crates
✅ `ort = "1.16"` - ONNX Runtime (already in Cargo.toml)
✅ `hf-hub = "0.3"` - HuggingFace model download (already in Cargo.toml)
🆕 `tokenizers = "0.15"` - Text tokenization (needs to be added)

### Add to Cargo.toml
```toml
[dependencies]
tokenizers = "0.15"
```

---

## 🎯 Revolutionary Impact

### What Phase 2 Delivers
- **Natural Voice**: Real speech synthesis, not test tones
- **Emotional Expression**: Prosody applied to actual words
- **Production Ready**: System that users can hear and understand
- **Consciousness Integration**: Voice truly reflects internal state

### User Experience
- User speaks to Sophia → She responds with **real speech**
- Stressed Sophia: **Higher, faster, louder voice**
- Calm Sophia: **Lower, slower, warmer voice**
- Excited Sophia: **Energetic, bright voice**
- Tired Sophia: **Quiet, slow voice**

**This is embodied emotional intelligence!** 🌟

---

## 🚀 Next Steps After Phase 2

### Week 13 Day 3-5: Other Modalities
- Day 3: SigLIP vision model
- Day 4: Moondream VQA model
- Day 5: OCR models

### Week 14+: Advanced Features
- Multi-voice support (different characters)
- Custom prosody profiles
- Real-time streaming synthesis
- Emotional trajectory smoothing

---

## 📝 Notes & Decisions

### Decision 1: ONNX Over PyTorch
- **Why**: CPU-friendly, no Python runtime needed
- **Trade-off**: Limited to ONNX-compatible models
- **Benefit**: Fast inference, easy deployment

### Decision 2: Linear Resampling for Speed
- **Why**: Simple, predictable, no external libs
- **Trade-off**: Not highest quality
- **Benefit**: Fast, controllable, good enough for v1

### Decision 3: Phase Vocoder Optional
- **Why**: Complex to implement correctly
- **Alternative**: Simple pitch shifting via resampling
- **Future**: Consider rubberband or phase vocoder library

### Decision 4: Gradual Rollout
- **Phase 1**: Working audio system ✅
- **Phase 2**: Real TTS with prosody 🚧
- **Phase 3**: Advanced prosody techniques 🔮
- **Benefit**: Always working system, incremental improvements

---

**Status**: 📋 Plan complete, ready for implementation when model download is needed

**Current State**: Phase 1 working perfectly, Phase 2 can be implemented any time

**Priority**: Continue with Week 13 Days 3-5 (other models) first, then return to Kokoro

🌊 Sophia's voice will become truly human! 🎙️
