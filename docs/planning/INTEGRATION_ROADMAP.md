# External Integration Roadmap

This document consolidates all external integration TODOs across the codebase, providing a clear roadmap for implementation.

## Status Legend
- **Stub**: Placeholder implementation exists
- **Planned**: Design exists, no implementation
- **Blocked**: Waiting on external dependency

---

## 1. OCR Integration (`perception/ocr.rs`)

**Status**: Stub

### Current Implementation
- `RtenOcr` stub exists but model loading not implemented
- `TesseractOcr` stub exists but command execution not implemented

### Required Work
1. Load rten/ocrs ONNX models (line 114)
2. Implement actual rten/ocrs inference (line 128)
3. Check tesseract binary availability (line 173)
4. Call tesseract via command line (line 192)

### Dependencies
- `rten` crate for ONNX runtime
- `ocrs` model files (need download/caching mechanism)
- `tesseract` binary for fallback

---

## 2. Semantic Vision (`perception/semantic_vision.rs`)

**Status**: Stub

### Current Implementation
- Model download stub exists
- ONNX inference stubs exist

### Required Work
1. Implement model download from HuggingFace Hub (line 413)
2. Implement ONNX inference for image encoding (line 425)
3. Implement question-conditioned inference (line 439)

### Dependencies
- `hf-hub` crate for model download
- ONNX runtime (`ort` or `rten`)
- CLIP/ViT model files

---

## 3. Text-to-Speech (`physiology/larynx.rs`)

**Status**: Stub

### Current Implementation
- Stub returns empty audio

### Required Work
1. Implement model download using hf-hub (line 316)
2. Load ONNX model using ort crate (line 331)

### Dependencies
- TTS model (e.g., Piper, XTTS)
- Audio output library

---

## 4. Database Integrations (`hdc/multi_database_integration.rs`)

**Status**: Planned

### Planned Databases
1. **Qdrant** - Sensory cortex (vector similarity search) (line 532)
2. **CozoDB** - Prefrontal cortex (graph reasoning) (line 536)
3. **LanceDB** - Long-term memory (columnar storage) (line 540)
4. **DuckDB** - Epistemic auditor (analytics) (line 544)

### Current State
All integrations are commented out with TODO markers.

### Required Work
- Define unified memory interface
- Implement adapters for each database
- Create migration/sync mechanisms

---

## 5. Long-term Memory (`hdc/long_term_memory.rs`)

**Status**: Stub

### Required Work
1. Implement Qdrant client integration (line 567)
2. Create collection management (line 581)
3. Implement vector search (line 594)

### Dependencies
- `qdrant-client` crate
- Running Qdrant instance

---

## 6. Proprioception (`physiology/proprioception.rs`)

**Status**: Stub

### Required Work
1. Implement actual disk reading with nix crate or similar (line 435)

### Dependencies
- System monitoring library
- Permissions for `/proc` access

---

## 7. Multimodal Perception (`perception/multi_modal.rs`)

**Status**: Stub

### Required Work
1. Implement actual projection using learned mapping (line 335)
2. Implement AST-based encoding for code (line 433)

### Dependencies
- Trained projection matrices
- AST parsing library (tree-sitter?)

---

## Implementation Priority

### High Priority (Core Functionality)
1. **Long-term Memory** - Required for persistent learning
2. **Semantic Vision** - Required for multimodal understanding
3. **OCR** - Required for document understanding

### Medium Priority (Enhanced Experience)
4. **Text-to-Speech** - Voice interaction
5. **Database Integrations** - Scalable memory

### Lower Priority (System Monitoring)
6. **Proprioception** - System awareness
7. **Multimodal Perception** - Code understanding

---

## Notes

- All stubs return reasonable defaults to prevent runtime errors
- Integration tests should mock external dependencies
- Consider feature flags to make integrations optional

---

*Last updated: 2025-01-09*
