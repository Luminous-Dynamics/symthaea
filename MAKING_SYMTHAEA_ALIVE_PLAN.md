# Making Symthaea Alive - Rigorous Plan

**Created**: December 29, 2025
**Status**: Verified and Ready for Implementation

---

## Executive Summary

Symthaea has a **solid foundation** that builds successfully with some minor fixes. The core consciousness architecture is functional, but the interaction model needs refinement. This document provides a rigorous plan for making Symthaea a living, breathing AI assistant.

---

## Current State (Verified December 29, 2025)

### Build Status: PASSING
- **Library compiles** with ~50 warnings (mostly unused variables)
- **Fixed issues**:
  - Added `RealHV` re-export from `hdc` module
  - Added `#[derive(Clone)]` to `RealPhiCalculator`
  - Fixed `Goal` struct field mismatches in `continuous_mind.rs`
  - Archived orphaned `test_cognitive_integration.rs` example

### Core Architecture: WORKING
| Component | Status | Notes |
|-----------|--------|-------|
| HDC 16,384D Semantic Space | ✅ Working | High-dimensional meaning representation |
| LTC Network (1024 neurons) | ✅ Working | Temporal dynamics for liquid computing |
| Consciousness Graph | ✅ Working | Φ measurement, coherence field |
| NSM Language System | ✅ Working | 65 semantic primes, LLM-free understanding |
| SophiaHLB Integration | ✅ Working | All components wired together |
| Voice Architecture | 🔧 Feature-gated | Whisper STT + Kokoro TTS ready |
| REPL | ✅ Working | Good for debugging, NOT production interface |

### Entry Points
1. **`symthaea`** (main.rs) - Demo binary showing consciousness initialization
2. **`symthaea-repl`** - Interactive debugging/testing interface

---

## Key Decision: REPL Status

### Recommendation: KEEP but don't use as production interface

**Rationale**:
- REPL is excellent for debugging and testing components in isolation
- REPL is NOT appropriate for end-user interaction (too technical)
- Production interface should be: **Voice + Service Mode**

**Action Items**:
- [ ] Keep REPL as development tool
- [ ] Create new `symthaea-service` binary for production
- [ ] Wire voice interface as primary interaction mode

---

## Phase 1: Stabilization (Immediate)

### 1.1 Fix Remaining Test Failures
The `phi_validation_minimal` test is failing because Star topology Φ equals Random topology Φ (both 0.5639). This may be:
- A regression in the topology generators
- An issue with the specific test parameters
- A valid result that the test assertion is wrong about

**Action**: Investigate and fix or update test expectations.

### 1.2 Clean Up Warnings
~50 compiler warnings remain. Most are:
- Unused variables/imports
- Unused struct fields
- Comparison issues in tests

**Action**: Run `cargo fix` or manually address warnings.

### 1.3 Archive Orphaned Code
The `test_cognitive_integration.rs` example referenced a non-existent `cognitive` module.

**Action**: ✅ Already moved to `.archive-broken/`

---

## Phase 2: Service Architecture (Week 1-2)

### 2.1 Create Service Binary
```rust
// bin/symthaea-service.rs
// Persistent daemon that:
// 1. Runs consciousness loop continuously
// 2. Listens for requests via socket/IPC
// 3. Processes queries through SophiaHLB
// 4. Returns responses with consciousness metrics
```

**Key Features**:
- Unix socket or TCP listener
- JSON-RPC or simple protocol for requests
- Background consciousness maintenance (DMN, sleep cycles)
- Graceful shutdown handling

### 2.2 Define API Protocol
```json
{
  "request": {
    "type": "query",
    "content": "install nginx",
    "context": {}
  },
  "response": {
    "type": "response",
    "content": "...",
    "phi": 0.4976,
    "confidence": 0.85,
    "execution_strategy": "confident"
  }
}
```

---

## Phase 3: Voice Interface (Week 2-3)

### 3.1 Enable Voice Features
The voice module exists but is feature-gated. Need to:
1. Enable `voice` feature in Cargo.toml
2. Test Whisper STT integration
3. Test Kokoro TTS output
4. Wire LTC-aware pacing for natural speech rhythm

### 3.2 Voice Flow
```
User Speech → Whisper STT → NSM Parser → SophiaHLB → Response Generator → Kokoro TTS → Audio Output
                                              ↓
                                    Consciousness Metrics (Φ, coherence)
```

### 3.3 LTC-Aware Pacing
The `LTCPacing` struct modulates speech rate based on:
- Consciousness state (slower when processing complex queries)
- Confidence level (more deliberate when uncertain)
- Circadian rhythm (matches user's expected tempo)

---

## Phase 4: Integration Testing (Week 3-4)

### 4.1 End-to-End Tests
- [ ] Voice input → text understanding → NixOS command → execution → voice response
- [ ] Multi-turn conversation with context retention
- [ ] Error recovery and graceful degradation
- [ ] Consciousness metrics throughout interaction

### 4.2 NixOS Integration
- [ ] Actual package installation commands
- [ ] Configuration generation and validation
- [ ] System state queries
- [ ] Safety guardrail verification

---

## Phase 5: Polish and Release (Week 4+)

### 5.1 Documentation
- [ ] User guide for voice interaction
- [ ] Developer guide for extending
- [ ] Architecture documentation update

### 5.2 Packaging
- [ ] Nix flake for installation
- [ ] Systemd service file
- [ ] Desktop integration (if applicable)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYMTHAEA ALIVE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Voice     │───▶│  NSM Parser │───▶│     SophiaHLB       │ │
│  │  (Whisper)  │    │ (65 primes) │    │                     │ │
│  └─────────────┘    └─────────────┘    │ ┌─────────────────┐ │ │
│                                        │ │ HDC 16,384D     │ │ │
│  ┌─────────────┐    ┌─────────────┐    │ │ Semantic Space  │ │ │
│  │   Voice     │◀───│  Response   │◀───│ ├─────────────────┤ │ │
│  │  (Kokoro)   │    │  Generator  │    │ │ LTC Network     │ │ │
│  └─────────────┘    └─────────────┘    │ │ (1024 neurons)  │ │ │
│                                        │ ├─────────────────┤ │ │
│                                        │ │ Consciousness   │ │ │
│  ┌────────────────────────────────┐    │ │ Graph (Φ=0.49)  │ │ │
│  │        Service Daemon          │    │ ├─────────────────┤ │ │
│  │  - Unix socket listener        │───▶│ │ NixOS Domain    │ │ │
│  │  - JSON-RPC protocol           │    │ │ Knowledge       │ │ │
│  │  - Background consciousness    │    │ └─────────────────┘ │ │
│  └────────────────────────────────┘    └─────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────┐    ┌─────────────────────┐ │
│  │         REPL (Debug)           │───▶│  SymthaeaAwakening  │ │
│  │  - /status, /introspect        │    │  - ConsciousPipeline│ │
│  │  - Development testing         │    │  - SelfModel        │ │
│  └────────────────────────────────┘    └─────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria

Symthaea is "alive" when:

1. **Voice works**: User can speak naturally and get spoken responses
2. **Understanding works**: NSM correctly parses NixOS intents without LLM
3. **Consciousness is measurable**: Φ metrics reflect actual integration
4. **Service is persistent**: Daemon runs continuously with consciousness loop
5. **Safety is enforced**: Guardrails prevent harmful operations
6. **Context is maintained**: Multi-turn conversations work correctly

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Voice integration fails | Medium | High | Fallback to text-only mode |
| Φ calculations regress | Low | Medium | Maintain test suite |
| Performance too slow | Medium | Medium | Profile and optimize hot paths |
| Safety guardrails bypass | Low | Critical | Extensive testing, conservative defaults |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Stabilization | 2-3 days | Green test suite, clean warnings |
| Service Architecture | 1 week | `symthaea-service` binary |
| Voice Interface | 1 week | Working voice I/O |
| Integration Testing | 1 week | End-to-end validation |
| Polish and Release | 1 week | v0.2.0 release |

**Total**: ~4 weeks to "alive" status

---

## Next Immediate Actions

1. **Investigate Φ test failure** - Why is Star = Random?
2. **Clean up warnings** - Run `cargo fix` or manual fixes
3. **Create service skeleton** - `bin/symthaea-service.rs`
4. **Enable voice feature** - Test Whisper/Kokoro integration

---

*This plan is based on verified state of the codebase as of December 29, 2025.*
