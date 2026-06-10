# Symthaea Comprehensive Improvement Plan

**Author**: Claude Opus 4.5
**Date**: 2026-01-28
**Version**: 1.0

This document provides detailed improvement plans for five key areas of Symthaea-HLB based on deep codebase analysis.

---

## Table of Contents

1. [Area 1: HDC+CfC Text-to-Speech (TTS)](#area-1-hdccfc-text-to-speech-tts)
2. [Area 2: Speech-to-Text (STT) Improvements](#area-2-speech-to-text-stt-improvements)
3. [Area 3: Rhythmic Speech / Rapping Capability](#area-3-rhythmic-speech--rapping-capability)
4. [Area 4: Stock Market Simulation](#area-4-stock-market-simulation)
5. [Area 5: HDC Dimension Unification](#area-5-hdc-dimension-unification)

---

## Area 1: HDC+CfC Text-to-Speech (TTS)

### Current State Analysis

**Existing Components:**
- `src/voice/mod.rs` (496 lines): LTCPacing struct with prosody parameters, placeholder sine-wave synthesis
- `crates/symthaea-stt/src/articulatory.rs` (1,127 lines): Full articulatory feature system (Voicing, Manner, Place, VowelHeight)
- `crates/symthaea-stt/src/articulatory_cfc.rs` (607 lines): Trained CfC model for articulatory feature detection
- `src/dynamics/cfc.rs`: Complete CfC network implementation with closed-form solutions

**Gap Analysis:**
- TTS is currently placeholder (sine waves, not actual speech)
- Articulatory system exists for STT but NOT for synthesis direction
- No vocoder implementation
- LTCPacing extracts prosody from network state but doesn't drive actual synthesis

### Implementation Plan

#### Phase 1: Articulatory TTS Foundation (2-3 weeks)

**1.1 Create Inverse Articulatory Mapper**

```
File: src/voice/articulatory_synthesizer.rs

Purpose: Map phonemes → articulatory features → control parameters
```

| Component | Input | Output | Implementation |
|-----------|-------|--------|----------------|
| Phoneme→Features | ARPABET phoneme | (Voicing, Manner, Place) | Reuse `ArticulatoryMapper` |
| Features→Parameters | Articulatory tuple | F1, F2, F3 formants | Acoustic-articulatory mapping table |
| Parameters→Trajectory | Formant targets | Time-varying formants | CfC interpolation |

**Key Code Structure:**
```rust
pub struct ArticulatorySynthesizer {
    // Reuse existing articulatory system
    mapper: ArticulatoryMapper,

    // CfC for smooth trajectory generation
    formant_cfc: CfCNetwork,  // From src/dynamics/cfc.rs

    // Prosody modulation from LTC
    prosody_ltc: UnifiedLTC,  // From src/unified_ltc.rs

    // Target formant frequencies per phoneme
    formant_targets: HashMap<String, (f32, f32, f32)>,  // F1, F2, F3
}

impl ArticulatorySynthesizer {
    pub fn synthesize_phoneme_sequence(
        &mut self,
        phonemes: &[Phoneme],
        pacing: &LTCPacing
    ) -> Vec<FormantFrame> {
        // 1. Convert phonemes to articulatory features
        // 2. Map features to formant targets
        // 3. Use CfC to interpolate smoothly (coarticulation)
        // 4. Apply pacing-based duration modulation
    }
}
```

**1.2 Formant Target Database**

Create formant frequency targets for each phoneme based on acoustic phonetics literature:

```rust
// src/voice/formant_targets.rs
pub const FORMANT_TARGETS: &[(&str, f32, f32, f32)] = &[
    // Vowels (F1, F2, F3 in Hz)
    ("IY", 270.0, 2290.0, 3010.0),   // "beat"
    ("IH", 390.0, 1990.0, 2550.0),   // "bit"
    ("EH", 530.0, 1840.0, 2480.0),   // "bet"
    ("AE", 660.0, 1720.0, 2410.0),   // "bat"
    ("AH", 520.0, 1190.0, 2390.0),   // "but"
    ("AA", 730.0, 1090.0, 2440.0),   // "bot"
    ("AO", 570.0, 840.0, 2410.0),    // "bought"
    ("UH", 440.0, 1020.0, 2240.0),   // "book"
    ("UW", 300.0, 870.0, 2240.0),    // "boot"
    // Consonants have rapid transitions, define as deltas
    // ...
];
```

**1.3 CfC Trajectory Generation**

Leverage existing `CfCNetwork` for smooth formant transitions:

```rust
impl ArticulatorySynthesizer {
    fn generate_formant_trajectory(
        &mut self,
        targets: &[(f32, f32, f32, f32)],  // (F1, F2, F3, duration)
        pacing: &LTCPacing
    ) -> Vec<FormantFrame> {
        let mut frames = Vec::new();

        for (i, (f1, f2, f3, dur)) in targets.iter().enumerate() {
            // Duration modulated by pacing.tau (slower tau = longer phonemes)
            let adjusted_dur = dur * (1.0 + (pacing.tau - 1.0) * 0.3);
            let num_frames = (adjusted_dur * SAMPLE_RATE / HOP_SIZE) as usize;

            // CfC interpolation with variable dt
            for frame_idx in 0..num_frames {
                let dt = adjusted_dur / num_frames as f32;
                let input = Array1::from_vec(vec![*f1, *f2, *f3]);
                let output = self.formant_cfc.forward(&input, dt);

                frames.push(FormantFrame {
                    f1: output[0],
                    f2: output[1],
                    f3: output[2],
                    energy: self.compute_energy(frame_idx, num_frames, pacing),
                    pitch: self.compute_pitch(frame_idx, num_frames, pacing),
                });
            }
        }
        frames
    }
}
```

#### Phase 2: Vocoder Integration (2-3 weeks)

**2.1 LPCNet-style Vocoder**

Implement a lightweight neural vocoder based on formant + excitation:

```rust
// src/voice/vocoder.rs
pub struct FormantVocoder {
    // Formant synthesis parameters
    formant_bandwidths: [f32; 3],  // B1, B2, B3

    // Excitation source
    glottal_pulse: GlottalPulseGenerator,
    noise_source: NoiseGenerator,

    // Output filter
    output_filter: BiquadFilter,

    sample_rate: u32,
}

impl FormantVocoder {
    pub fn synthesize(&mut self, frames: &[FormantFrame]) -> Vec<f32> {
        let mut audio = Vec::new();

        for frame in frames {
            // 1. Generate excitation (voiced: pulse train, unvoiced: noise)
            let excitation = if frame.voicing > 0.5 {
                self.glottal_pulse.generate(frame.pitch, HOP_SIZE)
            } else {
                self.noise_source.generate(HOP_SIZE)
            };

            // 2. Apply formant filter (cascade of resonators)
            let filtered = self.apply_formant_filter(&excitation, frame);

            // 3. Apply energy envelope
            let scaled = filtered.iter()
                .map(|s| s * frame.energy)
                .collect::<Vec<_>>();

            audio.extend(scaled);
        }

        audio
    }

    fn apply_formant_filter(&mut self, signal: &[f32], frame: &FormantFrame) -> Vec<f32> {
        // Cascade of 3 resonators (second-order IIR filters)
        let r1 = resonator_filter(signal, frame.f1, self.formant_bandwidths[0]);
        let r2 = resonator_filter(&r1, frame.f2, self.formant_bandwidths[1]);
        let r3 = resonator_filter(&r2, frame.f3, self.formant_bandwidths[2]);
        r3
    }
}
```

**2.2 Optional: ONNX Vocoder (HiFi-GAN/WaveGlow)**

For higher quality, integrate an ONNX vocoder:

```rust
// src/voice/neural_vocoder.rs (feature = "neural-vocoder")
pub struct NeuralVocoder {
    session: ort::Session,
    sample_rate: u32,
}

impl NeuralVocoder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = ort::Session::builder()?
            .with_model_from_file(model_path)?;
        Ok(Self { session, sample_rate: 22050 })
    }

    pub fn synthesize(&self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        let input = ort::Value::from_array(mel)?;
        let outputs = self.session.run(vec![input])?;
        Ok(outputs[0].try_extract()?.view().to_vec())
    }
}
```

#### Phase 3: HDC Semantic Binding (1-2 weeks)

**3.1 Semantic-Prosody Binding**

Bind semantic content to prosody using HDC:

```rust
// src/voice/semantic_prosody.rs
pub struct SemanticProsodyBinder {
    // Semantic encoder for text
    text_encoder: TextEncoder,  // From symthaea-core

    // Prosody prototype HVs
    prosody_prototypes: HashMap<ProsodyStyle, ContinuousHV>,

    // Binding role vectors
    role_content: ContinuousHV,
    role_emotion: ContinuousHV,
    role_emphasis: ContinuousHV,
}

impl SemanticProsodyBinder {
    pub fn encode_utterance(&self, text: &str, emotion: Emotion) -> BoundUtterance {
        // 1. Encode semantic content
        let content_hv = self.text_encoder.encode(text);

        // 2. Get emotion prosody prototype
        let emotion_hv = self.prosody_prototypes[&emotion.to_prosody()].clone();

        // 3. Bind: Utterance = Content ⊗ RoleContent ⊕ Emotion ⊗ RoleEmotion
        let bound = content_hv.bind(&self.role_content)
            .bundle(&[&emotion_hv.bind(&self.role_emotion)]);

        BoundUtterance {
            semantic: content_hv,
            prosody: emotion_hv,
            bound,
        }
    }

    pub fn derive_pacing(&self, bound: &BoundUtterance) -> LTCPacing {
        // Unbind prosody and map to LTCPacing parameters
        let prosody = bound.bound.unbind(&self.role_emotion);

        // Find nearest prosody prototype
        let (style, similarity) = self.find_nearest_prosody(&prosody);

        // Return corresponding pacing
        match style {
            ProsodyStyle::Calm => LTCPacing::calm(),
            ProsodyStyle::Excited => LTCPacing::excited(),
            ProsodyStyle::Focused => LTCPacing::focused(),
            ProsodyStyle::Custom(p) => p,
        }
    }
}
```

#### Phase 4: Full Pipeline Integration (1 week)

**4.1 Unified TTS Pipeline**

```rust
// src/voice/tts_pipeline.rs
pub struct HdcCfcTTS {
    // Text processing
    text_to_phonemes: TextToPhonemes,  // From symthaea-stt/lexicon.rs

    // Articulatory synthesis
    articulatory: ArticulatorySynthesizer,

    // Vocoder
    vocoder: FormantVocoder,

    // Semantic-prosody binding
    semantic_binder: SemanticProsodyBinder,

    // Configuration
    config: TTSConfig,
}

impl HdcCfcTTS {
    pub fn synthesize(&mut self, text: &str, emotion: Option<Emotion>) -> Result<Vec<f32>> {
        // 1. Semantic encoding and prosody derivation
        let emotion = emotion.unwrap_or(Emotion::Neutral);
        let bound = self.semantic_binder.encode_utterance(text, emotion);
        let pacing = self.semantic_binder.derive_pacing(&bound);

        // 2. Text to phonemes
        let phonemes = self.text_to_phonemes.convert(text)?;

        // 3. Phonemes to formant trajectories (CfC-based)
        let formant_frames = self.articulatory.synthesize_phoneme_sequence(&phonemes, &pacing);

        // 4. Vocoder synthesis
        let audio = self.vocoder.synthesize(&formant_frames);

        Ok(audio)
    }
}
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/voice/articulatory_synthesizer.rs` | Create | Phoneme→Formant mapping with CfC |
| `src/voice/formant_targets.rs` | Create | Acoustic phonetics database |
| `src/voice/vocoder.rs` | Create | Formant synthesis vocoder |
| `src/voice/neural_vocoder.rs` | Create | ONNX HiFi-GAN integration (optional) |
| `src/voice/semantic_prosody.rs` | Create | HDC semantic-prosody binding |
| `src/voice/tts_pipeline.rs` | Create | Full TTS pipeline |
| `src/voice/mod.rs` | Modify | Add module exports, integrate pipeline |
| `Cargo.toml` | Modify | Add `ort` dependency for ONNX (optional) |

### Expected Outcomes

- **Intelligible speech**: Formant synthesis produces recognizable speech
- **Emotional expression**: HDC prosody binding enables emotion-aware synthesis
- **Natural coarticulation**: CfC interpolation creates smooth transitions
- **LTC-driven pacing**: Network dynamics control speech rhythm
- **Extensibility**: ONNX vocoder path for future quality improvements

---

## Area 2: Speech-to-Text (STT) Improvements

### Current State Analysis

**Existing Pipeline:**
```
Audio → AudioFrontend (mel) → AudioProjector (LTC→HV16) → PhonemeDecoder (Hopfield) → Text
```

**Identified Weaknesses:**

| Issue | Location | Impact |
|-------|----------|--------|
| Limited HDC capacity | HV16 = 2,048 bits | Fine phonetic distinctions lost |
| Shallow context | 7-frame window | Long-range dependencies missed |
| Binary quantization | HV16 binary | Continuous phonetic info lost |
| No end-to-end training | Prototype bundling | Suboptimal representations |
| Fixed prototypes | Bootstrap only | No speaker adaptation |

### Implementation Plan

#### Phase 1: Upgrade to ContinuousHV (2 weeks)

**1.1 Replace HV16 with ContinuousHV in STT Pipeline**

```rust
// crates/symthaea-stt/src/hdc_v2.rs
use symthaea_core::hdc::unified_hv::ContinuousHV;

pub const STT_HDC_DIM: usize = 16_384;  // Match core dimension

pub struct SttHV(ContinuousHV);

impl SttHV {
    pub fn from_mel_frame(mel: &[f32], config: &MelEncodingConfig) -> Self {
        // Continuous encoding (not binary quantization)
        let mut values = vec![0.0f32; STT_HDC_DIM];

        // Position-sensitive encoding using random projection
        let projection = config.mel_projection_matrix();  // [STT_HDC_DIM x n_mels]

        for (i, row) in projection.iter().enumerate() {
            values[i] = row.iter()
                .zip(mel.iter())
                .map(|(w, m)| w * m)
                .sum();
        }

        SttHV(ContinuousHV::from_vec(values))
    }

    // Preserve backward compatibility
    pub fn to_hv16(&self) -> HV16 {
        self.0.to_binary()
    }
}
```

**1.2 Continuous Phoneme Prototypes**

```rust
// crates/symthaea-stt/src/prototype_v2.rs
pub struct ContinuousPrototypes {
    prototypes: HashMap<String, ContinuousHV>,
    dimension: usize,
}

impl ContinuousPrototypes {
    pub fn train_from_alignments(
        alignments: &[UtteranceAlignment],
        projector: &mut AudioProjectorV2
    ) -> Self {
        let mut accumulators: HashMap<String, (Vec<f32>, usize)> = HashMap::new();

        for alignment in alignments {
            for segment in &alignment.phoneme_segments {
                let audio = load_segment_audio(segment);
                let hv = projector.project_continuous(&audio);

                let entry = accumulators.entry(segment.phoneme.clone())
                    .or_insert((vec![0.0; STT_HDC_DIM], 0));

                // Accumulate (not bundle) for continuous average
                for (i, v) in hv.values().iter().enumerate() {
                    entry.0[i] += v;
                }
                entry.1 += 1;
            }
        }

        // Normalize and create prototypes
        let prototypes = accumulators.into_iter()
            .map(|(phoneme, (sum, count))| {
                let avg: Vec<f32> = sum.iter().map(|v| v / count as f32).collect();
                (phoneme, ContinuousHV::from_vec(avg).normalize())
            })
            .collect();

        Self { prototypes, dimension: STT_HDC_DIM }
    }
}
```

#### Phase 2: Hierarchical Temporal Encoding (2-3 weeks)

**2.1 Multi-Scale CfC Encoder**

```rust
// crates/symthaea-stt/src/hierarchical_encoder.rs
pub struct HierarchicalEncoder {
    // Frame level: 10ms windows
    frame_cfc: CfCNetwork,   // tau: 5-50ms

    // Phoneme level: ~50-100ms
    phoneme_cfc: CfCNetwork, // tau: 50-200ms

    // Word level: ~500ms
    word_cfc: CfCNetwork,    // tau: 200-1000ms

    // HDC binding across levels
    level_binders: [ContinuousHV; 3],  // Role vectors for binding
}

impl HierarchicalEncoder {
    pub fn encode_sequence(&mut self, frames: &[SttHV]) -> Vec<HierarchicalEncoding> {
        let mut encodings = Vec::new();

        // Reset CfC states
        self.frame_cfc.reset();
        self.phoneme_cfc.reset();
        self.word_cfc.reset();

        for (i, frame_hv) in frames.iter().enumerate() {
            let dt = 0.01;  // 10ms

            // Level 1: Frame processing
            let frame_input = Array1::from_vec(
                self.compress_hv(frame_hv, self.frame_cfc.config().input_dim)
            );
            let frame_state = self.frame_cfc.forward(&frame_input, dt);

            // Level 2: Phoneme processing (every 5 frames = 50ms)
            let phoneme_state = if i % 5 == 0 {
                let phoneme_input = frame_state.clone();
                self.phoneme_cfc.forward(&phoneme_input, dt * 5.0)
            } else {
                self.phoneme_cfc.read_state().unwrap()
            };

            // Level 3: Word processing (every 50 frames = 500ms)
            let word_state = if i % 50 == 0 {
                let word_input = phoneme_state.clone();
                self.word_cfc.forward(&word_input, dt * 50.0)
            } else {
                self.word_cfc.read_state().unwrap()
            };

            // Bind levels into hierarchical HV
            let frame_hv = ContinuousHV::from_vec(frame_state.to_vec());
            let phoneme_hv = ContinuousHV::from_vec(phoneme_state.to_vec());
            let word_hv = ContinuousHV::from_vec(word_state.to_vec());

            let bound = frame_hv.bind(&self.level_binders[0])
                .bundle(&[
                    &phoneme_hv.bind(&self.level_binders[1]),
                    &word_hv.bind(&self.level_binders[2]),
                ]);

            encodings.push(HierarchicalEncoding {
                frame: frame_hv,
                phoneme: phoneme_hv,
                word: word_hv,
                bound,
                timestamp: i as f32 * dt,
            });
        }

        encodings
    }
}
```

#### Phase 3: End-to-End Training (3-4 weeks)

**3.1 CTC Loss Integration**

```rust
// crates/symthaea-stt/src/ctc_training.rs
pub struct CtcTrainer {
    encoder: HierarchicalEncoder,
    output_projection: LinearLayer,  // Project to phoneme logits
    learning_rate: f32,
}

impl CtcTrainer {
    pub fn train_step(
        &mut self,
        audio: &[f32],
        target_phonemes: &[usize],
        blank_id: usize
    ) -> f32 {
        // Forward pass
        let mel_frames = extract_mel_features(audio);
        let hv_frames: Vec<SttHV> = mel_frames.iter()
            .map(|m| SttHV::from_mel_frame(m, &self.config))
            .collect();

        let encodings = self.encoder.encode_sequence(&hv_frames);

        // Project to phoneme logits
        let logits: Vec<Vec<f32>> = encodings.iter()
            .map(|enc| self.output_projection.forward(&enc.bound.values()))
            .collect();

        // CTC loss
        let (loss, gradients) = ctc_loss(&logits, target_phonemes, blank_id);

        // Backprop through output projection
        let hv_gradients = self.output_projection.backward(&gradients);

        // Backprop through CfC (BPTT for LTC networks)
        self.encoder.backward(&hv_gradients);

        loss
    }
}
```

**3.2 Differentiable HDC Operations**

To enable end-to-end training, make HDC operations differentiable:

```rust
// symthaea-core/src/hdc/differentiable.rs
pub struct DifferentiableContinuousHV {
    values: Vec<f32>,
    gradients: Option<Vec<f32>>,
}

impl DifferentiableContinuousHV {
    pub fn bind(&self, other: &Self) -> Self {
        // Element-wise multiplication (has simple gradient)
        let values: Vec<f32> = self.values.iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .collect();

        Self { values, gradients: None }
    }

    pub fn backward_bind(&self, other: &Self, upstream_grad: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // d(a*b)/da = b, d(a*b)/db = a
        let grad_self: Vec<f32> = upstream_grad.iter()
            .zip(other.values.iter())
            .map(|(g, b)| g * b)
            .collect();

        let grad_other: Vec<f32> = upstream_grad.iter()
            .zip(self.values.iter())
            .map(|(g, a)| g * a)
            .collect();

        (grad_self, grad_other)
    }

    pub fn bundle(hvs: &[&Self]) -> Self {
        // Sum and normalize (differentiable)
        let mut sum = vec![0.0f32; hvs[0].values.len()];
        for hv in hvs {
            for (i, v) in hv.values.iter().enumerate() {
                sum[i] += v;
            }
        }

        let norm: f32 = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
        let values: Vec<f32> = sum.iter().map(|v| v / norm).collect();

        Self { values, gradients: None }
    }
}
```

#### Phase 4: Online Adaptation (1-2 weeks)

**4.1 Speaker-Adaptive Prototypes**

```rust
// crates/symthaea-stt/src/adaptation_v2.rs
pub struct AdaptivePrototypes {
    base_prototypes: ContinuousPrototypes,
    speaker_offsets: HashMap<String, HashMap<String, ContinuousHV>>,  // speaker_id -> phoneme -> offset
    adaptation_rate: f32,
}

impl AdaptivePrototypes {
    pub fn adapt_online(
        &mut self,
        speaker_id: &str,
        recognition_result: &[RecognizedPhoneme],
        confidence_threshold: f32
    ) {
        let offsets = self.speaker_offsets
            .entry(speaker_id.to_string())
            .or_default();

        for result in recognition_result {
            if result.confidence > confidence_threshold {
                let base = &self.base_prototypes.prototypes[&result.phoneme];
                let observed = &result.observed_hv;

                // Compute offset as difference
                let diff = observed.subtract(base);

                // Update with EMA
                let offset = offsets.entry(result.phoneme.clone())
                    .or_insert(ContinuousHV::zero(base.dimension()));

                *offset = offset.scale(1.0 - self.adaptation_rate)
                    .add(&diff.scale(self.adaptation_rate));
            }
        }
    }

    pub fn get_adapted_prototype(&self, speaker_id: &str, phoneme: &str) -> ContinuousHV {
        let base = &self.base_prototypes.prototypes[phoneme];

        if let Some(offsets) = self.speaker_offsets.get(speaker_id) {
            if let Some(offset) = offsets.get(phoneme) {
                return base.add(offset).normalize();
            }
        }

        base.clone()
    }
}
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `crates/symthaea-stt/src/hdc_v2.rs` | Create | ContinuousHV-based STT encoding |
| `crates/symthaea-stt/src/prototype_v2.rs` | Create | Continuous prototype training |
| `crates/symthaea-stt/src/hierarchical_encoder.rs` | Create | Multi-scale CfC encoder |
| `crates/symthaea-stt/src/ctc_training.rs` | Create | End-to-end CTC training |
| `crates/symthaea-stt/src/adaptation_v2.rs` | Create | Online speaker adaptation |
| `symthaea-core/src/hdc/differentiable.rs` | Create | Differentiable HDC ops |
| `crates/symthaea-stt/src/lib.rs` | Modify | Export new modules |
| `crates/symthaea-stt/Cargo.toml` | Modify | Add training dependencies |

### Expected Outcomes

- **8x capacity increase**: 16,384-dim vs 2,048-bit vectors
- **Multi-scale context**: Frame + phoneme + word level information
- **Better accuracy**: End-to-end training optimizes full pipeline
- **Speaker adaptation**: Personalized recognition improves over time
- **Differentiable**: Full gradient flow enables neural integration

---

## Area 3: Rhythmic Speech / Rapping Capability

### Current State Analysis

**Existing Temporal/Rhythm Components:**
- `symthaea-core/src/hdc/temporal_binding.rs`: Theta-phase oscillations, 3-second binding window
- `crates/symthaea-sentinel/src/features.rs`: BPM-aware frequency bins (15-360 BPM)
- `crates/symthaea-sentinel/src/io.rs`: IOI (Inter-Onset Interval) variance computation
- `src/voice/mod.rs`: LTCPacing with rate, emphasis, breath_probability
- `src/dynamics/cfc.rs`: CfC networks with learnable tau

**Gap Analysis:**
- No beat synchronization mechanism
- No syllable-to-beat alignment
- No rhyme detection/encoding
- Pacing is continuous, not quantized to beat grid

### Implementation Plan

#### Phase 1: Beat Synchronization (1-2 weeks)

**1.1 Beat-Locked Tau Generator**

```rust
// src/voice/beat_sync.rs
pub struct BeatSyncEngine {
    bpm: f32,
    beat_phase: f32,        // 0.0 to 1.0 within beat
    beat_count: usize,
    subdivision: usize,     // 1=quarter, 2=eighth, 4=sixteenth

    // Phase-locked loop for beat tracking
    pll_phase: f32,
    pll_freq: f32,
    pll_gain: f32,
}

impl BeatSyncEngine {
    pub fn new(bpm: f32, subdivision: usize) -> Self {
        Self {
            bpm,
            beat_phase: 0.0,
            beat_count: 0,
            subdivision,
            pll_phase: 0.0,
            pll_freq: bpm / 60.0,
            pll_gain: 0.1,
        }
    }

    pub fn tick(&mut self, dt: f32) -> BeatPosition {
        // Update phase
        self.beat_phase += dt * self.bpm / 60.0;

        if self.beat_phase >= 1.0 {
            self.beat_phase -= 1.0;
            self.beat_count += 1;
        }

        BeatPosition {
            beat: self.beat_count,
            phase: self.beat_phase,
            subdivision_phase: (self.beat_phase * self.subdivision as f32) % 1.0,
            on_beat: self.beat_phase < 0.05 || self.beat_phase > 0.95,
            on_subdivision: (self.beat_phase * self.subdivision as f32) % 1.0 < 0.1,
        }
    }

    pub fn get_beat_tau(&self) -> f32 {
        // Tau = one beat duration (for LTC to sync with beat)
        60.0 / self.bpm
    }

    pub fn align_to_beat(&self, target_beat: usize, current_time: f32) -> f32 {
        // Return delay needed to hit target beat
        let beat_period = 60.0 / self.bpm;
        let target_time = target_beat as f32 * beat_period;
        (target_time - current_time).max(0.0)
    }
}
```

**1.2 Syllable-to-Beat Alignment**

```rust
// src/voice/syllable_align.rs
pub struct SyllableAligner {
    beat_sync: BeatSyncEngine,
    syllable_queue: VecDeque<Syllable>,
    current_beat: usize,
}

pub struct Syllable {
    pub text: String,
    pub phonemes: Vec<String>,
    pub stress: u8,           // 0, 1, or 2
    pub target_beat: Option<usize>,
    pub target_subdivision: Option<usize>,
}

impl SyllableAligner {
    pub fn align_lyrics(&mut self, lyrics: &[Line]) -> Vec<AlignedSyllable> {
        let mut aligned = Vec::new();
        let mut current_beat = 0;

        for line in lyrics {
            let syllables = self.syllabify_line(line);
            let beats_per_line = self.estimate_line_beats(&syllables);

            for (i, syl) in syllables.iter().enumerate() {
                // Stressed syllables land on beats
                let target = if syl.stress > 0 {
                    current_beat + (i * beats_per_line / syllables.len())
                } else {
                    // Unstressed on off-beats
                    current_beat + (i * beats_per_line / syllables.len())
                };

                aligned.push(AlignedSyllable {
                    syllable: syl.clone(),
                    beat: target,
                    subdivision: if syl.stress > 0 { 0 } else { 2 },  // On-beat vs off-beat
                    duration_beats: 1.0 / syllables.len() as f32 * beats_per_line as f32,
                });
            }

            current_beat += beats_per_line;
        }

        aligned
    }

    fn syllabify_line(&self, line: &Line) -> Vec<Syllable> {
        // Use CMU dictionary vowel nuclei to find syllable boundaries
        let mut syllables = Vec::new();

        for word in line.words.iter() {
            let phonemes = cmu_lookup(&word.text);
            let word_syllables = split_into_syllables(&phonemes);

            for (i, syl_phonemes) in word_syllables.iter().enumerate() {
                let stress = detect_stress(syl_phonemes);
                syllables.push(Syllable {
                    text: reconstruct_text(syl_phonemes),
                    phonemes: syl_phonemes.clone(),
                    stress,
                    target_beat: None,
                    target_subdivision: None,
                });
            }
        }

        syllables
    }
}
```

#### Phase 2: Rhyme HDC Encoding (1-2 weeks)

**2.1 Phonetic Rhyme Similarity**

```rust
// src/voice/rhyme_hdc.rs
pub struct RhymeEncoder {
    phoneme_hvs: HashMap<String, ContinuousHV>,
    position_hvs: Vec<ContinuousHV>,  // For positional encoding
}

impl RhymeEncoder {
    pub fn encode_word_ending(&self, phonemes: &[String], num_positions: usize) -> ContinuousHV {
        // Take last N phonemes (rhyme zone)
        let rhyme_phonemes = phonemes.iter()
            .rev()
            .take(num_positions)
            .collect::<Vec<_>>();

        // Bind with position vectors (end = position 0)
        let mut bound_hvs = Vec::new();
        for (pos, phoneme) in rhyme_phonemes.iter().enumerate() {
            if let Some(phon_hv) = self.phoneme_hvs.get(*phoneme) {
                let position_hv = &self.position_hvs[pos];
                bound_hvs.push(phon_hv.bind(position_hv));
            }
        }

        ContinuousHV::bundle(&bound_hvs.iter().collect::<Vec<_>>())
    }

    pub fn rhyme_score(&self, word1: &[String], word2: &[String]) -> f32 {
        let hv1 = self.encode_word_ending(word1, 3);  // Last 3 phonemes
        let hv2 = self.encode_word_ending(word2, 3);

        hv1.similarity(&hv2)
    }

    pub fn find_rhymes(&self, target: &[String], candidates: &[Vec<String>], threshold: f32) -> Vec<(usize, f32)> {
        let target_hv = self.encode_word_ending(target, 3);

        candidates.iter()
            .enumerate()
            .map(|(i, cand)| {
                let cand_hv = self.encode_word_ending(cand, 3);
                (i, target_hv.similarity(&cand_hv))
            })
            .filter(|(_, score)| *score > threshold)
            .collect()
    }
}
```

**2.2 Rhyme Scheme Detection**

```rust
// src/voice/rhyme_scheme.rs
pub struct RhymeSchemeDetector {
    encoder: RhymeEncoder,
    rhyme_threshold: f32,
}

impl RhymeSchemeDetector {
    pub fn detect_scheme(&self, lines: &[Vec<String>]) -> String {
        // Encode line endings
        let endings: Vec<ContinuousHV> = lines.iter()
            .map(|phonemes| self.encoder.encode_word_ending(phonemes, 3))
            .collect();

        // Cluster by similarity
        let mut scheme = String::new();
        let mut label_map: HashMap<usize, char> = HashMap::new();
        let mut next_label = 'A';

        for (i, hv) in endings.iter().enumerate() {
            // Check if rhymes with any previous line
            let mut matched = None;
            for (j, prev_hv) in endings[..i].iter().enumerate() {
                if hv.similarity(prev_hv) > self.rhyme_threshold {
                    matched = Some(j);
                    break;
                }
            }

            let label = if let Some(j) = matched {
                *label_map.get(&j).unwrap()
            } else {
                let l = next_label;
                label_map.insert(i, l);
                next_label = ((next_label as u8) + 1) as char;
                l
            };

            scheme.push(label);
        }

        scheme  // e.g., "AABB", "ABAB", "ABBA"
    }
}
```

#### Phase 3: Rap Synthesis Pipeline (2-3 weeks)

**3.1 RapSynthesizer**

```rust
// src/voice/rap.rs
pub struct RapSynthesizer {
    tts: HdcCfcTTS,
    beat_sync: BeatSyncEngine,
    syllable_aligner: SyllableAligner,
    rhyme_encoder: RhymeEncoder,
}

impl RapSynthesizer {
    pub fn synthesize(&mut self, lyrics: &[Line], bpm: f32) -> Vec<f32> {
        // Configure beat sync
        self.beat_sync = BeatSyncEngine::new(bpm, 4);  // 16th note subdivision

        // Align syllables to beats
        let aligned = self.syllable_aligner.align_lyrics(lyrics);

        // Generate audio
        let mut audio = Vec::new();
        let beat_duration = 60.0 / bpm;
        let sample_rate = 24000;

        for syl in &aligned {
            // Calculate syllable timing
            let start_time = syl.beat as f32 * beat_duration +
                            syl.subdivision as f32 * beat_duration / 4.0;
            let duration = syl.duration_beats * beat_duration;

            // Create pacing for this syllable
            let pacing = LTCPacing {
                rate: syl.syllable.phonemes.len() as f32 / duration,
                emphasis: if syl.syllable.stress > 0 { 1.3 } else { 0.9 },
                phrase_pause: 0.0,
                sentence_pause: 0.0,
                breath_probability: 0.0,
                emotional_valence: 0.5,
                arousal: 0.8,  // Rap is high energy
                tau: beat_duration,  // Lock to beat
            };

            // Synthesize syllable
            let syl_audio = self.tts.synthesize_with_pacing(
                &syl.syllable.phonemes.join(" "),
                &pacing
            )?;

            // Fit to duration (time-stretch if needed)
            let target_samples = (duration * sample_rate as f32) as usize;
            let fitted = time_stretch(&syl_audio, target_samples);

            // Insert at correct position
            let start_sample = (start_time * sample_rate as f32) as usize;
            while audio.len() < start_sample + fitted.len() {
                audio.push(0.0);
            }
            for (i, s) in fitted.iter().enumerate() {
                audio[start_sample + i] += s;
            }
        }

        audio
    }

    pub fn synthesize_with_beat(&mut self, lyrics: &[Line], beat_audio: &[f32], bpm: f32) -> Vec<f32> {
        let vocals = self.synthesize(lyrics, bpm)?;

        // Mix vocals with beat
        let mixed: Vec<f32> = vocals.iter()
            .zip(beat_audio.iter().cycle())
            .map(|(v, b)| v * 0.7 + b * 0.5)  // Vocals louder than beat
            .collect();

        mixed
    }
}
```

**3.2 Flow Styles**

```rust
// src/voice/flow.rs
pub enum FlowStyle {
    Straight,       // On-beat delivery
    Syncopated,     // Off-beat emphasis
    Triplet,        // 3-against-4 feel
    DoubleTimed,    // Twice as many syllables per beat
}

impl FlowStyle {
    pub fn apply_to_alignment(&self, aligned: &mut [AlignedSyllable], bpm: f32) {
        match self {
            FlowStyle::Straight => {
                // Already aligned to grid
            }
            FlowStyle::Syncopated => {
                for syl in aligned.iter_mut() {
                    if syl.syllable.stress == 0 {
                        // Push unstressed syllables slightly early
                        syl.subdivision = (syl.subdivision + 3) % 4;  // -1 in mod 4
                    }
                }
            }
            FlowStyle::Triplet => {
                // Convert to triplet grid
                for syl in aligned.iter_mut() {
                    syl.duration_beats *= 2.0 / 3.0;
                }
            }
            FlowStyle::DoubleTimed => {
                for syl in aligned.iter_mut() {
                    syl.duration_beats /= 2.0;
                }
            }
        }
    }
}
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/voice/beat_sync.rs` | Create | Beat synchronization engine |
| `src/voice/syllable_align.rs` | Create | Syllable-to-beat alignment |
| `src/voice/rhyme_hdc.rs` | Create | HDC rhyme encoding |
| `src/voice/rhyme_scheme.rs` | Create | Rhyme pattern detection |
| `src/voice/rap.rs` | Create | Full rap synthesis pipeline |
| `src/voice/flow.rs` | Create | Flow style variations |
| `src/voice/mod.rs` | Modify | Export rap modules |

### Expected Outcomes

- **Beat-locked delivery**: Syllables land precisely on beat grid
- **Rhyme awareness**: HDC encoding captures phonetic similarity
- **Flow variation**: Multiple rhythmic styles (straight, syncopated, double-time)
- **Integration with TTS**: Leverages existing HDC+CfC synthesis
- **Real-time capable**: CfC closed-form allows efficient computation

---

## Area 4: Stock Market Simulation

### Current State Analysis

**Existing Prediction Infrastructure:**
- `src/hdc/predictive_encoder.rs`: Attention-modulated prediction with error tracking
- `src/dynamics/cfc.rs`: CfC networks with irregular timestep handling
- `src/unified_ltc.rs`: UnifiedLTC with BPTT, Adam, Hebbian learning
- `src/hierarchical_cantor_ltc/`: Multi-scale temporal processing (7 levels, 1000ms→0.46ms)
- `src/intelligence/causal_discovery.rs`: Ensemble causal inference (71.3% accuracy on Tübingen)

**Gap Analysis:**
- No financial data loaders
- No market-specific regime detection
- No risk/portfolio management
- No real-time data feeds

### Implementation Plan

#### Phase 1: Market Data Infrastructure (1-2 weeks)

**1.1 OHLCV Data Structures**

```rust
// src/markets/data.rs
use chrono::{DateTime, Utc};

#[derive(Clone, Debug)]
pub struct OHLCV {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Clone, Debug)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub ohlcv: OHLCV,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

pub struct MarketDataLoader {
    cache: HashMap<String, Vec<OHLCV>>,
}

impl MarketDataLoader {
    pub fn load_csv(&mut self, symbol: &str, path: &Path) -> Result<()> {
        // Load historical data from CSV
    }

    pub fn load_yahoo(&mut self, symbol: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<()> {
        // Fetch from Yahoo Finance API
    }

    pub fn get_returns(&self, symbol: &str) -> Vec<f64> {
        let ohlcv = &self.cache[symbol];
        ohlcv.windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect()
    }

    pub fn get_log_returns(&self, symbol: &str) -> Vec<f64> {
        let ohlcv = &self.cache[symbol];
        ohlcv.windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }
}
```

**1.2 Technical Indicators**

```rust
// src/markets/indicators.rs
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        prices.windows(period)
            .map(|w| w.iter().sum::<f64>() / period as f64)
            .collect()
    }

    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = vec![prices[0]];

        for price in &prices[1..] {
            ema.push(alpha * price + (1.0 - alpha) * ema.last().unwrap());
        }

        ema
    }

    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        let changes: Vec<f64> = prices.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let gains: Vec<f64> = changes.iter().map(|c| c.max(0.0)).collect();
        let losses: Vec<f64> = changes.iter().map(|c| (-c).max(0.0)).collect();

        let avg_gains = Self::sma(&gains, period);
        let avg_losses = Self::sma(&losses, period);

        avg_gains.iter()
            .zip(avg_losses.iter())
            .map(|(g, l)| {
                if *l == 0.0 { 100.0 }
                else { 100.0 - 100.0 / (1.0 + g / l) }
            })
            .collect()
    }

    pub fn bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, period);

        let std: Vec<f64> = prices.windows(period)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / period as f64;
                (w.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / period as f64).sqrt()
            })
            .collect();

        let upper: Vec<f64> = sma.iter().zip(std.iter()).map(|(s, d)| s + std_dev * d).collect();
        let lower: Vec<f64> = sma.iter().zip(std.iter()).map(|(s, d)| s - std_dev * d).collect();

        (lower, sma, upper)
    }
}
```

#### Phase 2: HDC Market Encoding (2 weeks)

**2.1 Market State Encoder**

```rust
// src/markets/hdc_encoder.rs
use symthaea_core::hdc::unified_hv::ContinuousHV;

pub struct MarketHDCEncoder {
    // Feature basis vectors
    price_momentum_hv: ContinuousHV,
    volatility_hv: ContinuousHV,
    volume_hv: ContinuousHV,
    trend_hv: ContinuousHV,

    // Regime prototypes
    regime_prototypes: HashMap<MarketRegime, ContinuousHV>,

    // Quantization levels
    num_levels: usize,
    level_hvs: Vec<ContinuousHV>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    HighVolatility,
    LowVolatility,
    Trending,
    MeanReverting,
}

impl MarketHDCEncoder {
    pub fn encode_snapshot(&self, snapshot: &MarketSnapshot, indicators: &IndicatorValues) -> ContinuousHV {
        // Encode price momentum (returns)
        let momentum_level = self.quantize(indicators.returns_5d, -0.1, 0.1);
        let momentum_encoded = self.level_hvs[momentum_level].bind(&self.price_momentum_hv);

        // Encode volatility (realized vol)
        let vol_level = self.quantize(indicators.volatility_20d, 0.0, 0.5);
        let vol_encoded = self.level_hvs[vol_level].bind(&self.volatility_hv);

        // Encode volume (relative to average)
        let vol_ratio = snapshot.ohlcv.volume / indicators.avg_volume_20d;
        let volume_level = self.quantize(vol_ratio, 0.5, 2.0);
        let volume_encoded = self.level_hvs[volume_level].bind(&self.volume_hv);

        // Encode trend (price relative to SMA)
        let trend_ratio = snapshot.ohlcv.close / indicators.sma_50;
        let trend_level = self.quantize(trend_ratio, 0.9, 1.1);
        let trend_encoded = self.level_hvs[trend_level].bind(&self.trend_hv);

        // Bundle all features
        ContinuousHV::bundle(&[
            &momentum_encoded,
            &vol_encoded,
            &volume_encoded,
            &trend_encoded,
        ])
    }

    pub fn detect_regime(&self, state_hv: &ContinuousHV) -> (MarketRegime, f32) {
        self.regime_prototypes.iter()
            .map(|(regime, proto)| (*regime, state_hv.similarity(proto)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    fn quantize(&self, value: f64, min: f64, max: f64) -> usize {
        let normalized = ((value - min) / (max - min)).clamp(0.0, 0.999);
        (normalized * self.num_levels as f64) as usize
    }
}
```

#### Phase 3: CfC Price Dynamics Model (2-3 weeks)

**3.1 Multi-Horizon Predictor**

```rust
// src/markets/predictor.rs
use crate::dynamics::cfc::{CfCNetwork, CfCNetworkConfig};

pub struct MarketPredictor {
    // Multi-horizon CfC networks
    short_term: CfCNetwork,   // tau: 1-5 days
    medium_term: CfCNetwork,  // tau: 5-20 days
    long_term: CfCNetwork,    // tau: 20-60 days

    // HDC encoder
    encoder: MarketHDCEncoder,

    // Regime-dependent tau adjustment
    regime_tau_modifiers: HashMap<MarketRegime, f32>,

    // Prediction history for calibration
    predictions: VecDeque<PredictionRecord>,
}

pub struct PredictionRecord {
    pub timestamp: DateTime<Utc>,
    pub horizon: usize,  // days
    pub predicted_return: f64,
    pub predicted_volatility: f64,
    pub confidence: f64,
    pub actual_return: Option<f64>,
}

impl MarketPredictor {
    pub fn predict(&mut self, snapshot: &MarketSnapshot, indicators: &IndicatorValues) -> MarketPrediction {
        // Encode current state
        let state_hv = self.encoder.encode_snapshot(snapshot, indicators);

        // Detect regime
        let (regime, regime_confidence) = self.encoder.detect_regime(&state_hv);

        // Adjust tau based on regime
        let tau_mod = *self.regime_tau_modifiers.get(&regime).unwrap_or(&1.0);

        // Compress HV for CfC input
        let input = self.compress_hv(&state_hv, 64);
        let input_array = Array1::from_vec(input);

        // Multi-horizon predictions
        let short_pred = self.short_term.forward(&input_array, 1.0 * tau_mod);
        let medium_pred = self.medium_term.forward(&input_array, 5.0 * tau_mod);
        let long_pred = self.long_term.forward(&input_array, 20.0 * tau_mod);

        MarketPrediction {
            symbol: snapshot.symbol.clone(),
            timestamp: snapshot.ohlcv.timestamp,
            regime,
            regime_confidence,
            short_term: HorizonPrediction {
                horizon_days: 1,
                expected_return: short_pred[0] as f64,
                volatility: short_pred[1].abs() as f64,
                confidence: self.calibrated_confidence(1),
            },
            medium_term: HorizonPrediction {
                horizon_days: 5,
                expected_return: medium_pred[0] as f64,
                volatility: medium_pred[1].abs() as f64,
                confidence: self.calibrated_confidence(5),
            },
            long_term: HorizonPrediction {
                horizon_days: 20,
                expected_return: long_pred[0] as f64,
                volatility: long_pred[1].abs() as f64,
                confidence: self.calibrated_confidence(20),
            },
        }
    }

    pub fn train_step(&mut self, history: &[MarketSnapshot], indicators_history: &[IndicatorValues]) -> f32 {
        let mut total_loss = 0.0;

        // Train on historical sequences
        for i in 20..history.len() - 20 {
            let snapshot = &history[i];
            let indicators = &indicators_history[i];

            // Targets: future returns
            let target_1d = (history[i + 1].ohlcv.close - snapshot.ohlcv.close) / snapshot.ohlcv.close;
            let target_5d = (history[i + 5].ohlcv.close - snapshot.ohlcv.close) / snapshot.ohlcv.close;
            let target_20d = (history[i + 20].ohlcv.close - snapshot.ohlcv.close) / snapshot.ohlcv.close;

            let state_hv = self.encoder.encode_snapshot(snapshot, indicators);
            let input = Array1::from_vec(self.compress_hv(&state_hv, 64));

            // Train each horizon
            let loss_1d = self.short_term.train_step(
                &input,
                &Array1::from_vec(vec![target_1d as f32, 0.0]),
                1.0,
                0.001
            )?;

            let loss_5d = self.medium_term.train_step(
                &input,
                &Array1::from_vec(vec![target_5d as f32, 0.0]),
                5.0,
                0.001
            )?;

            let loss_20d = self.long_term.train_step(
                &input,
                &Array1::from_vec(vec![target_20d as f32, 0.0]),
                20.0,
                0.001
            )?;

            total_loss += loss_1d + loss_5d + loss_20d;
        }

        total_loss / (history.len() - 40) as f32
    }

    fn calibrated_confidence(&self, horizon: usize) -> f64 {
        // Use historical prediction accuracy for calibration
        let relevant: Vec<&PredictionRecord> = self.predictions.iter()
            .filter(|p| p.horizon == horizon && p.actual_return.is_some())
            .collect();

        if relevant.len() < 30 {
            return 0.5;  // Not enough data
        }

        // Compute direction accuracy
        let correct = relevant.iter()
            .filter(|p| {
                let pred_sign = p.predicted_return.signum();
                let actual_sign = p.actual_return.unwrap().signum();
                pred_sign == actual_sign
            })
            .count();

        correct as f64 / relevant.len() as f64
    }
}
```

#### Phase 4: Causal Analysis Integration (1-2 weeks)

**4.1 Market Causal Discovery**

```rust
// src/markets/causal.rs
use crate::intelligence::causal_discovery::{CausalDiscovery, CausalEdge};

pub struct MarketCausalAnalyzer {
    discovery: CausalDiscovery,
    causal_graph: HashMap<String, Vec<CausalEdge>>,
}

impl MarketCausalAnalyzer {
    pub fn discover_causal_structure(
        &mut self,
        symbols: &[String],
        data: &HashMap<String, Vec<OHLCV>>
    ) -> CausalGraph {
        let mut edges = Vec::new();

        // Analyze pairwise causality
        for i in 0..symbols.len() {
            for j in 0..symbols.len() {
                if i == j { continue; }

                let returns_i = compute_returns(&data[&symbols[i]]);
                let returns_j = compute_returns(&data[&symbols[j]]);

                // Granger causality test
                if let Some(direction) = self.discovery.granger_causality(
                    &returns_i, &returns_j, 5  // 5 lag periods
                ) {
                    edges.push(CausalEdge {
                        from: symbols[i].clone(),
                        to: symbols[j].clone(),
                        strength: direction.strength,
                        lag: direction.lag,
                    });
                }
            }
        }

        // Also analyze sector/macro causality
        // VIX → equity returns
        // Treasury yields → rate-sensitive sectors
        // etc.

        CausalGraph { edges }
    }

    pub fn causal_adjusted_prediction(
        &self,
        target: &str,
        predictions: &HashMap<String, MarketPrediction>,
        graph: &CausalGraph
    ) -> MarketPrediction {
        let base_pred = predictions[target].clone();

        // Find causes of target
        let causes: Vec<&CausalEdge> = graph.edges.iter()
            .filter(|e| e.to == target)
            .collect();

        // Adjust prediction based on causal signals
        let mut adjusted_return = base_pred.short_term.expected_return;

        for cause in causes {
            if let Some(cause_pred) = predictions.get(&cause.from) {
                // Propagate causal effect with lag and strength
                let causal_contribution = cause_pred.short_term.expected_return
                    * cause.strength
                    * (-cause.lag as f64 * 0.1).exp();  // Decay with lag

                adjusted_return += causal_contribution;
            }
        }

        MarketPrediction {
            short_term: HorizonPrediction {
                expected_return: adjusted_return,
                ..base_pred.short_term
            },
            ..base_pred
        }
    }
}
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/markets/mod.rs` | Create | Market simulation module root |
| `src/markets/data.rs` | Create | OHLCV data structures and loaders |
| `src/markets/indicators.rs` | Create | Technical indicator calculations |
| `src/markets/hdc_encoder.rs` | Create | HDC market state encoding |
| `src/markets/predictor.rs` | Create | CfC multi-horizon prediction |
| `src/markets/causal.rs` | Create | Causal analysis for markets |
| `src/markets/backtest.rs` | Create | Backtesting framework |
| `src/lib.rs` | Modify | Export markets module |
| `Cargo.toml` | Modify | Add chrono, reqwest for data fetching |

### Expected Outcomes

- **HDC regime detection**: Identify bull/bear/sideways regimes via similarity
- **Multi-horizon prediction**: CfC networks predict 1/5/20 day returns
- **Causal adjustment**: Use cross-asset causality to improve predictions
- **Calibrated confidence**: Historical accuracy for uncertainty quantification
- **Extensible**: Framework supports additional assets, indicators, strategies

### Important Caveats

⚠️ **This is NOT financial advice.** Backtested results do not guarantee future performance. Markets are complex, non-stationary systems. Use for research/education only.

---

## Area 5: HDC Dimension Unification

### Current State Analysis

**Dimension Map:**

| System | Dimension | Type | Storage |
|--------|-----------|------|---------|
| STT (HV16) | 2,048 bits | Binary | 256 bytes |
| Core | 16,384 | Continuous | 65.5 KB |
| Extended | 32,768 | Continuous | 131 KB |
| Embeddings | 768-1,024 | Float32 | 3-4 KB |

**Current Bridge (HV16 → Core):**
- 8x expansion via BLAKE3 hashing
- Majority voting for compression
- Similarity-preserving but lossy

**Issues:**
1. STT operates at 1/8th capacity of core
2. Bridge adds latency and complexity
3. Mixed binary/continuous complicates training
4. No gradients flow through bridge

### Implementation Plan

#### Phase 1: Unified Dimension Configuration (1 week)

**1.1 Centralized Configuration**

```rust
// symthaea-core/src/hdc/config.rs
#[derive(Clone, Copy, Debug)]
pub struct HdcConfig {
    pub dimension: usize,
    pub num_levels: usize,      // For quantization
    pub sparse_density: f32,    // For sparse operations
}

impl HdcConfig {
    pub const STANDARD: Self = Self {
        dimension: 16_384,
        num_levels: 16,
        sparse_density: 0.1,
    };

    pub const COMPACT: Self = Self {
        dimension: 4_096,
        num_levels: 8,
        sparse_density: 0.2,
    };

    pub const EXTENDED: Self = Self {
        dimension: 32_768,
        num_levels: 32,
        sparse_density: 0.05,
    };
}

// Global configuration (set at startup)
static HDC_CONFIG: OnceLock<HdcConfig> = OnceLock::new();

pub fn set_hdc_config(config: HdcConfig) {
    HDC_CONFIG.set(config).expect("HDC config already set");
}

pub fn hdc_dim() -> usize {
    HDC_CONFIG.get().map(|c| c.dimension).unwrap_or(16_384)
}
```

#### Phase 2: Migrate STT to ContinuousHV (2-3 weeks)

**2.1 Replace HV16 with ConfigurableHV**

```rust
// crates/symthaea-stt/src/hdc_unified.rs
use symthaea_core::hdc::{ContinuousHV, hdc_dim};

pub type SttHV = ContinuousHV;  // Alias for clarity

// Remove HV16 from public API
// Keep internally for legacy compatibility if needed
#[deprecated(note = "Use SttHV (ContinuousHV) instead")]
pub type HV16 = crate::hdc::HV16;

impl SttHV {
    pub fn from_mel(mel: &[f32], projection: &MelProjection) -> Self {
        let values = projection.project(mel);
        ContinuousHV::from_vec(values)
    }
}
```

**2.2 Update AudioProjector**

```rust
// crates/symthaea-stt/src/audio_v2.rs
pub struct AudioProjectorV2 {
    config: AudioConfig,
    mel_projection: MelProjection,
    ltc: LtcCell,
    dimension: usize,
}

impl AudioProjectorV2 {
    pub fn new(config: AudioConfig) -> Self {
        let dimension = hdc_dim();  // Use global config
        Self {
            config,
            mel_projection: MelProjection::new(config.n_mels, dimension),
            ltc: LtcCell::new(LtcConfig {
                hidden_size: 64,
                ..Default::default()
            }),
            dimension,
        }
    }

    pub fn project(&mut self, audio: &[f32]) -> Vec<SttHV> {
        let frontend = AudioFrontend::new(self.config.clone());
        let mel_frames = frontend.extract_features(audio);

        mel_frames.iter()
            .map(|mel| {
                // LTC processing
                let ltc_out = self.ltc.step(mel, self.config.hop_duration);

                // Project to full dimension
                SttHV::from_mel(&ltc_out, &self.mel_projection)
            })
            .collect()
    }
}
```

#### Phase 3: Differentiable Bridge (2 weeks)

**3.1 Learned Projection (Replaces Hard-Coded Expansion)**

```rust
// symthaea-core/src/hdc/projection.rs
pub struct LearnedProjection {
    // Learned projection matrix
    weights: Array2<f32>,  // [output_dim x input_dim]

    // Optimizer state
    momentum: Array2<f32>,
    velocity: Array2<f32>,

    input_dim: usize,
    output_dim: usize,
}

impl LearnedProjection {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let weights = Array2::from_shape_fn(
            (output_dim, input_dim),
            |_| (rand::random::<f32>() - 0.5) * 2.0 * scale
        );

        Self {
            weights,
            momentum: Array2::zeros((output_dim, input_dim)),
            velocity: Array2::zeros((output_dim, input_dim)),
            input_dim,
            output_dim,
        }
    }

    pub fn forward(&self, input: &ContinuousHV) -> ContinuousHV {
        let input_arr = Array1::from_vec(input.values.clone());
        let output_arr = self.weights.dot(&input_arr);
        ContinuousHV::from_vec(output_arr.to_vec())
    }

    pub fn backward(&mut self, input: &ContinuousHV, grad_output: &[f32], lr: f32) -> Vec<f32> {
        let input_arr = Array1::from_vec(input.values.clone());
        let grad_arr = Array1::from_vec(grad_output.to_vec());

        // Gradient w.r.t. weights: outer(grad_output, input)
        let grad_weights = grad_arr.clone().insert_axis(ndarray::Axis(1))
            .dot(&input_arr.insert_axis(ndarray::Axis(0)));

        // Gradient w.r.t. input: W^T @ grad_output
        let grad_input = self.weights.t().dot(&grad_arr);

        // Adam update
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        self.momentum = &self.momentum * beta1 + &grad_weights * (1.0 - beta1);
        self.velocity = &self.velocity * beta2 + &grad_weights.mapv(|x| x * x) * (1.0 - beta2);

        let m_hat = &self.momentum / (1.0 - beta1);
        let v_hat = &self.velocity / (1.0 - beta2);

        self.weights = &self.weights - &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + eps)) * lr;

        grad_input.to_vec()
    }
}
```

**3.2 Bidirectional Bridge**

```rust
// symthaea-core/src/hdc/bridge.rs
pub struct BidirectionalBridge {
    // STT → Core projection (upscale)
    stt_to_core: LearnedProjection,

    // Core → STT projection (downscale)
    core_to_stt: LearnedProjection,

    // Reconstruction loss weight
    recon_weight: f32,
}

impl BidirectionalBridge {
    pub fn new(stt_dim: usize, core_dim: usize) -> Self {
        Self {
            stt_to_core: LearnedProjection::new(stt_dim, core_dim),
            core_to_stt: LearnedProjection::new(core_dim, stt_dim),
            recon_weight: 0.1,
        }
    }

    pub fn encode(&self, stt_hv: &SttHV) -> ContinuousHV {
        self.stt_to_core.forward(&ContinuousHV::from_vec(stt_hv.values().to_vec()))
    }

    pub fn decode(&self, core_hv: &ContinuousHV) -> SttHV {
        let decoded = self.core_to_stt.forward(core_hv);
        SttHV::from_vec(decoded.values)
    }

    pub fn train_step(&mut self, stt_hv: &SttHV, target_core_hv: &ContinuousHV, lr: f32) -> f32 {
        // Forward
        let encoded = self.encode(stt_hv);
        let reconstructed = self.decode(&encoded);

        // Losses
        let encoding_loss = mse(&encoded.values, &target_core_hv.values);
        let recon_loss = mse(&reconstructed.values(), stt_hv.values());
        let total_loss = encoding_loss + self.recon_weight * recon_loss;

        // Backward (simplified)
        let grad_encoded = grad_mse(&encoded.values, &target_core_hv.values);
        let grad_stt = self.stt_to_core.backward(
            &ContinuousHV::from_vec(stt_hv.values().to_vec()),
            &grad_encoded,
            lr
        );

        total_loss
    }
}
```

#### Phase 4: Update All Consumers (1-2 weeks)

**4.1 Migration Checklist**

| Module | Current | Target | Migration Steps |
|--------|---------|--------|-----------------|
| `symthaea-stt/hdc.rs` | HV16 (2048) | SttHV (configurable) | Replace type alias, update ops |
| `symthaea-stt/audio.rs` | HV16 projector | AudioProjectorV2 | New projector class |
| `symthaea-stt/phoneme.rs` | HV16 prototypes | ContinuousHV | Update resonator |
| `src/perception/audio.rs` | HV16 bridge | Direct ContinuousHV | Remove expansion |
| `src/hdc/predictive_encoder.rs` | RealHV (16384) | ContinuousHV | Rename type |
| `symthaea-core/temporal_binding.rs` | RealHV (2048) | ContinuousHV (config) | Use hdc_dim() |

**4.2 Compatibility Layer**

```rust
// symthaea-core/src/hdc/compat.rs
// For gradual migration

pub trait HVCompat {
    fn to_continuous(&self) -> ContinuousHV;
    fn from_continuous(hv: &ContinuousHV) -> Self;
}

impl HVCompat for HV16 {
    fn to_continuous(&self) -> ContinuousHV {
        // Expand each bit to f32
        let values: Vec<f32> = self.words.iter()
            .flat_map(|word| {
                (0..128).map(move |i| {
                    if (word >> i) & 1 == 1 { 1.0 } else { -1.0 }
                })
            })
            .collect();

        // Expand to target dimension
        let target_dim = hdc_dim();
        if values.len() == target_dim {
            ContinuousHV::from_vec(values)
        } else {
            // Use learned projection or simple interpolation
            interpolate_to_dim(&values, target_dim)
        }
    }

    fn from_continuous(hv: &ContinuousHV) -> Self {
        // Threshold to binary
        let values = if hv.values.len() > 2048 {
            downsample(&hv.values, 2048)
        } else {
            hv.values.clone()
        };

        let mut words = [0u128; 16];
        for (i, v) in values.iter().enumerate() {
            if *v > 0.0 {
                words[i / 128] |= 1u128 << (i % 128);
            }
        }

        HV16 { words }
    }
}
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `symthaea-core/src/hdc/config.rs` | Create | Centralized dimension config |
| `symthaea-core/src/hdc/projection.rs` | Create | Learned projection layer |
| `symthaea-core/src/hdc/bridge.rs` | Create | Bidirectional bridge |
| `symthaea-core/src/hdc/compat.rs` | Create | Compatibility layer |
| `crates/symthaea-stt/src/hdc_unified.rs` | Create | Unified STT HDC |
| `crates/symthaea-stt/src/audio_v2.rs` | Create | Updated audio projector |
| `crates/symthaea-stt/src/hdc.rs` | Modify | Deprecate HV16, add exports |
| `symthaea-core/src/hdc/mod.rs` | Modify | Export new modules |

### Expected Outcomes

- **Unified dimension**: All systems use configurable dimension (default 16,384)
- **No capacity loss**: STT operates at full dimension
- **Differentiable**: Gradients flow through projection layers
- **Backward compatible**: Compat layer supports gradual migration
- **Configurable**: Support for compact (4K), standard (16K), extended (32K) modes

---

## Implementation Priority Matrix

| Area | Impact | Effort | Priority | Dependencies |
|------|--------|--------|----------|--------------|
| **5. Dimension Unification** | High | Medium | 1 (First) | None |
| **2. STT Improvements** | High | High | 2 | Area 5 |
| **1. TTS** | High | High | 3 | Area 5 |
| **3. Rapping** | Medium | Medium | 4 | Areas 1, 5 |
| **4. Stock Market** | Medium | Medium | 5 | Area 5 |

**Rationale:** Dimension unification is foundational—all other improvements benefit from unified HDC space. STT and TTS are core capabilities. Rapping builds on TTS. Stock market is independent but benefits from unified HDC.

---

## Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Dimension Unification | 3-4 weeks | Config system, learned projection, compat layer |
| Phase 2: STT Upgrades | 4-5 weeks | ContinuousHV STT, hierarchical encoder, CTC training |
| Phase 3: TTS Pipeline | 5-6 weeks | Articulatory synth, vocoder, semantic binding |
| Phase 4: Rapping | 3-4 weeks | Beat sync, syllable alignment, rhyme encoding |
| Phase 5: Market Simulation | 4-5 weeks | Data infrastructure, HDC encoding, CfC predictor |

**Total: 19-24 weeks** for full implementation (can be parallelized)

---

## Conclusion

This improvement plan provides a comprehensive roadmap for enhancing Symthaea's capabilities across five key areas. The architecture leverages Symthaea's unique strengths:

1. **HDC's compositional algebra** for semantic binding
2. **CfC's closed-form solutions** for efficient temporal processing
3. **LTC's adaptive dynamics** for variable-rate phenomena
4. **Hopfield networks** for associative memory

The implementations are designed to be modular, testable, and backward-compatible where possible. Each area can be developed incrementally while maintaining system stability.
