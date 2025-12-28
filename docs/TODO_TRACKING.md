# Symthaea TODO Tracking

**Last Updated**: December 29, 2025
**Total Items**: 46
**Status**: Documented and categorized

---

## Summary by Category

| Category | Count | Priority |
|----------|-------|----------|
| Model Integration (ONNX/HuggingFace) | 12 | Medium - Deferred |
| Database Integration | 7 | Medium - Deferred |
| Algorithm Implementation | 10 | High - Core functionality |
| Test Calibration | 4 | Medium - Quality |
| Feature Stubs | 8 | Low - Future work |
| Minor Improvements | 5 | Low - Polish |

---

## Category 1: Model Integration (ONNX/HuggingFace)

These TODOs relate to integrating ML models via ONNX Runtime and HuggingFace Hub.
**Status**: Deferred until `audio` feature is fully activated.

### perception/semantic_vision.rs (7 items)
- Line 19: Activate ONNX Runtime import
- Line 23: Activate HuggingFace Hub import
- Line 119: Download CLIP model from HuggingFace Hub
- Line 132: Implement actual ONNX inference for CLIP
- Line 197: Download SigLIP model from HuggingFace Hub
- Line 209: Implement ONNX inference for SigLIP
- Line 223: Implement ONNX inference with question conditioning

### perception/ocr.rs (4 items)
- Line 114: Load rten/ocrs models
- Line 128: Implement rten/ocrs inference
- Line 173: Check if tesseract command exists
- Line 192: Call Tesseract via command line

### physiology/larynx.rs (2 items)
- Line 316: Implement model download using hf-hub
- Line 331: Load ONNX model using ort crate

---

## Category 2: Database Integration

These TODOs relate to the multi-database consciousness architecture.
**Status**: Deferred until database features (qdrant, lance, duck) are activated.

### hdc/multi_database_integration.rs (4 items)
- Line 532: Integrate Qdrant for sensory cortex
- Line 536: Integrate CozoDb for prefrontal cortex
- Line 540: Integrate LanceDb for long-term memory
- Line 544: Integrate DuckDb for epistemic auditor

### hdc/long_term_memory.rs (3 items)
- Line 567: Implement actual Qdrant client integration
- Line 581: Implement store with qdrant_client
- Line 594: Implement retrieve with qdrant_client

---

## Category 3: Algorithm Implementation (High Priority)

These TODOs represent core algorithmic work that should be addressed.

### perception/multi_modal.rs (5 items)
- **Line 30**: Replace HDC placeholder with proper implementation from hdc module
  - *Action*: Use `RealHV` from `hdc::real_hv` module
- **Line 191**: Implement actual projection using learned mapping
- **Line 228**: Implement Johnson-Lindenstrauss random projection
- **Line 250**: Implement proper text encoding
- **Line 273**: Implement AST-based code encoding

### hdc/collective_consciousness.rs (2 items)
- **Line 487**: Implement full clustering coefficient calculation
- **Line 496**: Implement full shortest path calculation

### observability/counterfactual_reasoning.rs (1 item)
- **Line 216**: Implement proper uncertainty propagation

### sleep_cycles.rs (1 item)
- **Line 246**: Use resonator networks to find recurring patterns
  - *Note*: `phi_resonant.rs` now provides this capability

### physiology/proprioception.rs (1 item)
- **Line 435**: Implement actual disk reading with nix crate

---

## Category 4: Test Calibration

Tests that are ignored pending calibration work.

### hdc/phi_tier_tests.rs (3 items)
- **Line 84**: `#[ignore]` - State generation needs calibration for IIT-compliant partition sampling
- **Line 150**: `#[ignore]` - State generation needs calibration for IIT-compliant partition sampling
- **Line 225**: `#[ignore]` - Heuristic algorithm needs tuning to match exact within 30%

### brain/daemon.rs (1 item)
- **Line 525**: `#[ignore]` - Fix memory ID selection logic

---

## Category 5: Feature Stubs

Placeholder implementations for future features.

### swarm.rs (2 items)
- Line 150: Actually send via gossipsub (libp2p integration)
- Line 173: Wait for responses and aggregate

### observability/streaming_causal.rs (2 items)
- Line 289: Time-based eviction if configured
- Line 375: Alert 3: Rapid event sequence detection

### observability/pattern_library.rs (2 items)
- Line 378: Track pattern deviations
- Line 437: Track pattern deviations

### observability/causal_intervention.rs (1 item)
- Line 200: Handle multiple intervention nodes properly

### safety/amygdala.rs (1 item)
- Line 312: Phase 2: Broadcast "Cortisol Spike" to Endocrine Core

---

## Category 6: Minor Improvements

Small enhancements and tracking improvements.

### language/conversation.rs (2 items)
- Line 549: Extract word_trace from dynamic generation
- Line 2542: Track actual sentence form

### language/nix_knowledge_provider.rs (1 item)
- Line 617: Parse configuration.nix to extract system state

### consciousness/recursive_improvement/mod.rs (1 item)
- Line 14: Extract future modules from core.rs

---

## Resolution Strategy

### Immediate (This Sprint)
1. Fix `perception/multi_modal.rs` HDC integration (use existing `RealHV`)
2. Enable resonator network usage in `sleep_cycles.rs`

### Short-term (Next Month)
1. Address algorithm implementations in Category 3
2. Calibrate phi_tier_tests

### Medium-term (Next Quarter)
1. Model integration when `audio` feature is production-ready
2. Database integration when backend features activated

### Long-term (Backlog)
1. Feature stubs (swarm, advanced observability)
2. Minor improvements

---

## How to Update This Document

When resolving a TODO:
1. Remove the TODO comment from the code
2. Update this document (move to "Resolved" section below)
3. Add resolution date and commit hash

## Resolved Items

| Date | File | Line | Resolution | Commit |
|------|------|------|------------|--------|
| *None yet* | | | | |

---

*This document is auto-generated from codebase analysis. Run `grep -rn "TODO\|FIXME" src/` to verify.*
