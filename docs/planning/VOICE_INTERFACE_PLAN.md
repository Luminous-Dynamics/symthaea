# Voice Interface Plan: Kokoro TTS + whisper-rs STT

**Status**: Planning Complete
**Target**: v0.3.0
**Dependencies**: Cargo features for optional voice

## Overview

Natural voice interface for Symthaea using:
- **STT (Speech-to-Text)**: whisper-rs (Rust bindings for whisper.cpp)
- **TTS (Text-to-Speech)**: Kokoro (via ONNX runtime)

## Why These Choices

### whisper-rs for STT
| Factor | whisper-rs | Alternatives |
|--------|-----------|--------------|
| **Language** | Rust bindings to C++ | Python (Vosk), native Rust (limited) |
| **Accuracy** | OpenAI Whisper-level | Variable |
| **Speed** | ~10x realtime on CPU | Varies |
| **Offline** | Yes | Most yes |
| **Models** | tiny→large (39M→1.5B) | Fixed |

### Kokoro for TTS
| Factor | Kokoro | Alternatives |
|--------|--------|--------------|
| **Quality** | Very natural | espeak (robotic), Piper (good) |
| **Speed** | Fast ONNX inference | Variable |
| **Size** | ~80MB model | Variable |
| **Rust** | Via `ort` crate | Native bindings rare |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VOICE INTERFACE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐    │
│   │   Audio     │    │   Conversation  │    │   Audio     │    │
│   │   Input     │───▶│     Engine      │───▶│   Output    │    │
│   │ (whisper-rs)│    │  (existing)     │    │  (Kokoro)   │    │
│   └─────────────┘    └─────────────────┘    └─────────────┘    │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐    │
│   │  Whisper    │    │  LTC Temporal   │    │   ONNX      │    │
│   │   Model     │    │   Dynamics      │    │  Runtime    │    │
│   │ (tiny/base) │    │  (flow pacing)  │    │  (Kokoro)   │    │
│   └─────────────┘    └─────────────────┘    └─────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## LTC Integration for Voice

The LTC flow_state will influence voice pacing:

```rust
// In voice output
let flow = ltc_snapshot.flow_state;
let speech_rate = if flow > 0.7 {
    1.1  // Peak flow = slightly faster, confident
} else if flow > 0.4 {
    1.0  // Normal flow = natural pace
} else {
    0.9  // Warming up = slower, thoughtful
};

let pause_duration = if ltc_snapshot.phi_trend > 0.0 {
    200  // Rising Φ = shorter pauses, engaged
} else {
    400  // Falling Φ = longer pauses, reflective
};
```

## Cargo Dependencies

```toml
[features]
voice = ["whisper-rs", "ort", "cpal", "hound"]

[dependencies]
# STT: whisper-rs (C++ bindings)
whisper-rs = { version = "0.12", optional = true }

# TTS: ONNX Runtime for Kokoro
ort = { version = "2.0", optional = true }

# Audio I/O
cpal = { version = "0.15", optional = true }
hound = { version = "3.5", optional = true }
```

## Implementation Plan

### Phase 1: STT Foundation
1. Add whisper-rs dependency
2. Create `VoiceInput` struct with model loading
3. Implement audio capture via cpal
4. Basic transcription → Conversation::respond()

```rust
pub struct VoiceInput {
    ctx: WhisperContext,
    model_size: ModelSize,
    language: Option<String>,
}

impl VoiceInput {
    pub fn new(model_size: ModelSize) -> Result<Self>;
    pub fn transcribe(&self, audio: &[f32]) -> Result<String>;
    pub fn start_listening(&self) -> Receiver<String>;
}
```

### Phase 2: TTS Output
1. Add Kokoro ONNX model
2. Create `VoiceOutput` struct
3. Implement text → audio synthesis
4. Audio playback via cpal

```rust
pub struct VoiceOutput {
    session: ort::Session,
    sample_rate: u32,
}

impl VoiceOutput {
    pub fn new() -> Result<Self>;
    pub fn speak(&self, text: &str) -> Result<Vec<f32>>;
    pub fn speak_with_rate(&self, text: &str, rate: f32) -> Result<Vec<f32>>;
}
```

### Phase 3: LTC-Aware Pacing
1. Feed flow_state to speech rate
2. Add natural pauses based on Φ trend
3. Modulate pitch based on emotional valence

### Phase 4: Full Integration
1. Create `VoiceConversation` wrapper
2. Handle interruption gracefully
3. Implement wake word detection (optional)
4. Add voice activity detection (VAD)

## File Structure

```
src/voice/
├── mod.rs           # Re-exports, feature gate
├── input.rs         # VoiceInput (whisper-rs)
├── output.rs        # VoiceOutput (Kokoro/ONNX)
├── conversation.rs  # VoiceConversation wrapper
├── audio.rs         # cpal audio I/O utilities
└── models/          # Model loading utilities
```

## Model Downloads

Models should be downloaded on first use:

```rust
// ~/.local/share/symthaea/models/
//   whisper-tiny.bin (39MB)
//   kokoro-v0_19.onnx (~80MB)

pub fn ensure_models() -> Result<ModelPaths> {
    let whisper_path = download_if_missing(
        "whisper-tiny.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
    )?;
    let kokoro_path = download_if_missing(
        "kokoro-v0_19.onnx",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx"
    )?;
    Ok(ModelPaths { whisper: whisper_path, kokoro: kokoro_path })
}
```

## Usage Example

```rust
use symthaea::voice::{VoiceConversation, VoiceConfig};

fn main() -> Result<()> {
    let config = VoiceConfig {
        whisper_model: ModelSize::Tiny,
        use_ltc_pacing: true,
        language: None, // Auto-detect
    };

    let mut voice = VoiceConversation::new(config)?;

    // Start voice loop
    voice.run()?;

    // Or manual control
    while let Some(text) = voice.listen()? {
        let response = voice.respond(&text)?;
        voice.speak(&response)?;
    }

    Ok(())
}
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| STT Latency | <500ms | After speech end |
| TTS Latency | <200ms | First audio out |
| Total Turn | <1s | End-to-end |
| Memory | <500MB | With models loaded |
| CPU (idle) | <5% | When listening |

## Testing Plan

1. **Unit tests**: Mock audio, test transcription/synthesis
2. **Integration tests**: Full voice loop with recorded audio
3. **Performance tests**: Latency measurement
4. **LTC tests**: Verify pacing changes with flow state

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Large model downloads | Tiny model (39MB) default, optional larger |
| ONNX runtime issues | Fallback to espeak-ng |
| Audio device conflicts | cpal device enumeration |
| Whisper accuracy | Allow language hint, base model option |

## Next Steps

1. [ ] Add Cargo features and dependencies
2. [ ] Implement VoiceInput with whisper-rs
3. [ ] Implement VoiceOutput with Kokoro
4. [ ] Create VoiceConversation wrapper
5. [ ] Add LTC pacing integration
6. [ ] Test on real hardware
7. [ ] Document usage and model download

## References

- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust bindings for whisper.cpp
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - Lightweight TTS model
- [ort](https://github.com/pykeio/ort) - ONNX Runtime for Rust
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O
