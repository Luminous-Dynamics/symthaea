# Unified HDC Architecture

**Date**: January 2026
**Status**: In Progress

---

## Overview

Symthaea uses multiple HDC (Hyperdimensional Computing) implementations across different modules. This document describes the unified architecture centered on **16,384-dimensional continuous vectors** and the upgrade paths for modules still using legacy formats.

---

## Current HDC Implementations

### 1. symthaea-core: ContinuousHV (Unified Standard)

**Location**: `crates/symthaea-core/src/hdc/unified_hv.rs`
**Dimension**: **16,384** (HDC_DIMENSION constant)
**Type**: Continuous f32 vectors

```rust
pub const HDC_DIMENSION: usize = 16_384;

pub struct ContinuousHV {
    values: Vec<f32>,  // 16,384-element continuous vector
}
```

**Features**:
- Continuous values enable smooth interpolation and gradient-based operations
- Supports bundling, binding, permutation, and similarity computation
- Used by: PhonemeHdcCodec, Voice module, Markets module

---

### 2. symthaea-stt: HV16 (2,048-bit Binary)

**Location**: `crates/symthaea-stt/src/hdc.rs`
**Dimension**: 2,048 bits (stored as 16 x u128)
**Type**: Binary (0/1 or bipolar +1/-1)

```rust
pub const HDC_DIM: usize = 2048;

pub struct HV16 {
    pub words: [u128; 16],  // 2048-bit binary vector
}
```

**Features**:
- Compact binary representation (256 bytes per vector)
- BLAKE3-based deterministic random projection
- XOR binding, majority voting bundling
- Used by: LiquidHDC, cetacean acoustic analysis, speech recognition

---

### 3. Brain/Actor Model: SharedHdcVector (10,000-dim Bipolar)

**Location**: `src/brain/actor_model.rs`
**Dimension**: 10,000
**Type**: Bipolar i8 (-1/+1)

```rust
pub type SharedHdcVector = Arc<Vec<i8>>;  // 10,000-element bipolar
```

**Features**:
- Hierarchical 5-layer semantic encoding
- Zero-copy sharing with Arc
- Used by: Attention arena, hippocampus memory

---

## Bridging Infrastructure (STT)

The STT crate already has comprehensive bridging code for interoperability:

### HV16 to ContinuousHV Conversion

```rust
// Expand from 2,048 to 16,384 dimensions
impl HV16 {
    /// Expand to continuous f32 representation at core dimension (16,384)
    pub fn to_core_continuous(&self) -> Vec<f32> {
        // Each bit expands to 8 bits in output (8x expansion)
        // Bit 1 -> +1.0, Bit 0 -> -1.0
    }

    /// Compress from core continuous back to HV16
    pub fn from_core_continuous(values: &[f32]) -> Self {
        // Groups of 8 values vote for 1 output bit
        // Positive sum -> 1, Negative sum -> 0
    }
}
```

### Similarity Preservation

The expansion/compression is **similarity-preserving**:

```rust
#[test]
fn test_expansion_preserves_similarity() {
    let a = HV16::random("A");
    let b = HV16::random("B");

    // Original similarity
    let sim_original = a.similarity(&b);

    // Expanded similarity
    let a_exp = a.expand_to_core();
    let b_exp = b.expand_to_core();
    let sim_expanded = hamming_similarity(&a_exp, &b_exp);

    // Should match within tolerance
    assert!((sim_original - sim_expanded).abs() < 0.01);
}
```

---

## STT Upgrade Path

### Phase 1: Interop Layer (CURRENT)

The STT crate can already interoperate with symthaea-core via the conversion methods:

```rust
// In STT code
let stt_vector = HV16::random("phoneme_AE");

// Convert to core format for unified operations
let core_vector: Vec<f32> = stt_vector.to_core_continuous();

// Use with ContinuousHV operations
let core_hv = ContinuousHV::from_slice(&core_vector);
let similarity = core_hv.similarity(&other_core_hv);

// Convert back if needed
let back_to_stt = HV16::from_core_continuous(&core_hv.values());
```

### Phase 2: Native ContinuousHV (OPTIONAL)

For maximum integration, STT could migrate to native ContinuousHV:

**Pros**:
- Unified codebase, no conversion overhead
- Access to all ContinuousHV features (lerp, resonance, etc.)
- Consistent dimension across all modules

**Cons**:
- 8x memory increase (256 bytes -> 64KB per vector)
- Binary operations (XOR) require thresholding
- Existing cetacean models would need retraining

**Recommendation**: Keep HV16 for STT-specific operations (low latency, small footprint) but use `to_core_continuous()` when integrating with voice/cognitive modules.

### Phase 3: Hybrid Architecture

The recommended architecture uses both representations:

```text
                    ┌─────────────────────────────────────┐
                    │       UNIFIED HDC BUS (16,384)      │
                    │         ContinuousHV Format          │
                    └────────────────┬────────────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
       ▼                             ▼                             ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│  Voice/TTS   │            │   Markets    │            │  STT Crate   │
│  ContinuousHV│            │  ContinuousHV│            │    HV16      │
│   (native)   │            │   (native)   │            │ (+ bridge)   │
└──────────────┘            └──────────────┘            └──────────────┘
       │                             │                             │
       ▼                             ▼                             ▼
  PhonemeHDC                   MarketHV                    LiquidHDC
  RhymeEncoder                 Pattern Rec.            Cetacean Grammar
  CognitiveBridge              Trend Pred.              Whistle Decode
```

---

## Integration Examples

### Voice + Cognitive Loop (COMPLETE)

The cognitive bridge connects CfC predictions to voice synthesis:

```rust
// In cognitive_bridge.rs
pub struct CognitivePacing {
    pub base_pacing: LTCPacing,
    pub confidence: f32,           // From prediction error
    pub word_emphasis: HashMap<String, f32>,  // From attention
    pub highlighted_concepts: Vec<String>,     // Detected primitives
}

// Usage
let bridge = CognitiveVoiceBridge::new();
let pacing = bridge.update(
    &cfc_output,       // CfC neural state
    tau,               // Time constant
    prediction_error,  // Uncertainty
    attention_state,   // Focus weights
    detected_primitives,
);

// Voice synthesis uses HDC-driven coarticulation
let synth = ArticulatorySynthesizer::new(config);
synth.set_phoneme_codec(hdc_codec);  // Uses ContinuousHV
```

### TTS + HDC Coarticulation (COMPLETE)

Phoneme transitions use HDC similarity for natural blending:

```rust
// In phoneme_hdc.rs (ContinuousHV, 16,384-dim)
let similarity = codec.similarity("AE", "EY");  // 0.72 (similar vowels)
let tau_modifier = codec.coarticulation_weight("AE", "EY");  // 1.28

// In articulatory_synthesizer.rs
fn synthesize_transition(&self, from: &str, to: &str) {
    let blend = self.hdc_blend_factor(from, to);  // HDC-driven
    let tau = self.base_tau * self.hdc_tau_modifier(from, to);
    // Smooth blending based on phonetic similarity
}
```

### STT Integration (BRIDGED)

STT can bridge to the unified bus when needed:

```rust
// In STT code
let whistle_hv = HV16::random("whistle_pattern");
let click_hv = HV16::random("click_pattern");

// For integration with TTS/Voice
let whistle_core = whistle_hv.to_core_continuous();
let click_core = click_hv.to_core_continuous();

// Compare with voice phoneme encodings
let phoneme_codec = PhonemeHdcCodec::new();
let ae_hv = phoneme_codec.encode("AE").unwrap();

// Cross-modal similarity (whistle sounds like "AE"?)
let cross_sim = ContinuousHV::from_slice(&whistle_core)
    .similarity(&ae_hv);
```

---

## Module Status

| Module | HDC Type | Dimension | Status |
|--------|----------|-----------|--------|
| symthaea-core | ContinuousHV | 16,384 | Unified Standard |
| Voice/TTS | ContinuousHV | 16,384 | Migrated |
| Markets | ContinuousHV | 16,384 | Migrated |
| Rhyme | ContinuousHV | 16,384 | Migrated |
| Cognitive Bridge | ContinuousHV | 16,384 | NEW |
| STT | HV16 + Bridge | 2,048 (8x expand) | Interop Ready |
| Brain/Actor | SharedHdcVector | 10,000 | Legacy |

---

## Migration Checklist

### Completed
- [x] PhonemeHdcCodec using ContinuousHV (16,384-dim)
- [x] RhymeEncoder delegating to PhonemeHdcCodec
- [x] ArticulatorySynthesizer with HDC coarticulation
- [x] Markets using ContinuousHV
- [x] CognitiveVoiceBridge connecting CfC to voice
- [x] STT bridging infrastructure (to_core_continuous)

### Pending
- [ ] Brain/Actor model upgrade to 16,384-dim (optional)
- [ ] STT native ContinuousHV (optional, not recommended)
- [ ] SIMD optimization for ContinuousHV operations
- [ ] GPU acceleration for large-batch HDC operations

---

## Performance Notes

### Memory Usage

| Type | Per-Vector | 1000 Vectors |
|------|------------|--------------|
| HV16 (2,048-bit) | 256 bytes | 250 KB |
| ContinuousHV (16,384 f32) | 64 KB | 62.5 MB |
| SharedHdcVector (10,000 i8) | 10 KB | 9.77 MB |

### Computation Time (approximate)

| Operation | HV16 | ContinuousHV |
|-----------|------|--------------|
| Similarity | ~1 μs | ~10 μs |
| Binding | ~0.5 μs | ~5 μs |
| Bundling (10 vecs) | ~5 μs | ~50 μs |
| Expansion/Compression | N/A | ~20 μs |

The 8x memory and ~10x computation cost of ContinuousHV is justified by:
1. Smooth interpolation for voice synthesis
2. Gradient-friendly learning
3. Unified cross-module similarity

---

## References

- `crates/symthaea-core/src/hdc/unified_hv.rs` - ContinuousHV implementation
- `crates/symthaea-stt/src/hdc.rs` - HV16 with bridge code
- `src/voice/phoneme_hdc.rs` - Phoneme HDC codec
- `src/voice/cognitive_bridge.rs` - CfC-voice integration
- `src/markets/market_state.rs` - Market HDC patterns
